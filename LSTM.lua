
local LSTM = {}
function LSTM.create(input_size, rnn_size, n)
  -- there will be 2*n+1 inputs  ???
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x   解决方法
  for L = 1,n do  -- n = num layers
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]  --L*2+1 解决方法
    local prev_c = inputs[L*2]
    -- the input to this layer

    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      input_size_L = rnn_size
    end
    --解决
    -- x = outputs[(L-1)*2] 
    -- input_size_L = rnn_size

    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    -- previous cell state contribution
    local c_forget = nn.CMulTable()({forget_gate, prev_c})
    -- input contribution
    local c_input = nn.CMulTable()({in_gate, in_transform})
    -- next cell state
    local next_c = nn.CAddTable()({c_forget, c_input})
    -- gated cells form the output
    local c_transform = nn.Tanh()(next_c)
    local next_h = nn.CMulTable()({out_gate, c_transform})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  return nn.gModule(inputs, outputs)  -- inputs={x, prev_c, prev_h}  outputs = {next_c, next_h}
end

return LSTM

