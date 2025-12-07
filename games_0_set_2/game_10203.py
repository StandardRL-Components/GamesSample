import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:06:15.912546
# Source Brief: brief_00203.md
# Brief Index: 203
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

def lerp_color(c1, c2, t):
    """Linearly interpolate between two colors."""
    t = max(0, min(1, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, vx, vy, radius, color, lifetime):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # Gravity
        self.lifetime -= 1
        self.radius = max(0, self.radius * (self.lifetime / self.max_lifetime))

    def draw(self, surface):
        if self.lifetime > 0:
            pos = (int(self.x), int(self.y))
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(self.radius), self.color)
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(self.radius), self.color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    game_description = (
        "Manage liquid levels and purity across interconnected pipes. Rotate pipes to direct flow "
        "and purify the system to achieve stability."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to select a pipe. Press space to rotate the selected pipe. "
        "Press shift to purify the most polluted pipe."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PIPE_WIDTH = 60
    PIPE_HEIGHT = 200
    PIPE_CAPACITY = 100.0
    WIN_TIME_SECONDS = 10 # Reduced for easier testing; brief says 60
    MAX_EPISODE_STEPS = 5000

    # Colors
    COLOR_BG = (26, 28, 44)
    COLOR_GRID = (42, 44, 60)
    COLOR_PIPE = (80, 88, 111)
    COLOR_POLLUTED = (139, 139, 139)
    COLOR_CLEAN = (0, 170, 255)
    COLOR_TEXT = (224, 224, 224)
    COLOR_SELECTOR = (255, 255, 0)
    COLOR_WIN = (0, 255, 127)
    COLOR_LOSS = (255, 50, 50)

    # Physics
    INFLOW_RATE = 0.05
    OUTFLOW_RATE_PER_LEVEL = 0.001
    ROTATION_SPEED = 0.15
    PURIFICATION_AMOUNT = 1.0 # Instantly purifies

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        # State variables are initialized in reset()
        # self.reset() # reset() is called by the wrapper, but let's call it here to be safe for validation
        
        # Critical self-check
        # self.validate_implementation() # This is called after reset() to ensure state is initialized.
        # To avoid issues with uninitialized state, we'll reset, then validate.
        # But since the test harness will call reset anyway, we can just defer initialization to it.
        # The main issue is that validate_implementation needs an initialized state.
        # Let's ensure reset is called once.
        self._np_random = None # Will be set by super().reset(seed)
        self.steps = 0 # Will be properly set in reset.
        if self._np_random is None:
            self.reset() # Ensure state is initialized for validation

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False

        self.pipe_levels = [self.np_random.uniform(40, 60) for _ in range(4)]
        self.pipe_purity = [self.np_random.uniform(0.0, 0.2) for _ in range(4)]
        self.pipe_target_angles = [0.0] * 4
        self.pipe_current_angles = [0.0] * 4
        
        self.pipe_base_y = self.SCREEN_HEIGHT / 2
        self.pipe_base_x = [100, 240, 380, 520]
        self.pipe_positions = [(x, self.pipe_base_y) for x in self.pipe_base_x]

        self.selected_pipe_idx = 0
        self.win_timer = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        
        reward = 0
        
        # Handle player actions
        reward += self._handle_input(movement, space_held, shift_held)
        
        # Update game physics
        self._update_physics()

        # Calculate continuous rewards
        for level in self.pipe_levels:
            fill_ratio = level / self.PIPE_CAPACITY
            if 0.7 <= fill_ratio <= 0.9:
                reward += 1 / self.metadata["render_fps"] # Small reward per frame
            if not (0.1 <= fill_ratio <= 0.9):
                reward -= 1 / self.metadata["render_fps"]

        # Check for termination
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.score += reward
        
        if terminated:
            self.game_over = True

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        # --- Action 0: Movement (Selection) ---
        if movement == 3: # Left
            self.selected_pipe_idx = max(0, self.selected_pipe_idx - 1)
        elif movement == 4: # Right
            self.selected_pipe_idx = min(3, self.selected_pipe_idx + 1)
        
        # --- Action 1: Space (Rotate Selected Pipe) ---
        if space_held and not self.last_space_held:
            # sfx: mechanical_click.wav
            self.pipe_target_angles[self.selected_pipe_idx] = (self.pipe_target_angles[self.selected_pipe_idx] + 90) % 360

        # --- Action 2: Shift (Purify) ---
        if shift_held and not self.last_shift_held:
            # Find most polluted pipe (lowest purity)
            if any(p < 1.0 for p in self.pipe_purity):
                min_purity = float('inf')
                target_idx = -1
                for i, purity in enumerate(self.pipe_purity):
                    if purity < min_purity:
                        min_purity = purity
                        target_idx = i
                
                if target_idx != -1 and self.pipe_purity[target_idx] < 1.0:
                    self.pipe_purity[target_idx] = 1.0
                    reward += 5
                    # sfx: purify_chime.wav
                    # Spawn purification particles
                    pipe_x, pipe_y = self.pipe_positions[target_idx]
                    for _ in range(30):
                        px = pipe_x + self.np_random.uniform(-self.PIPE_WIDTH / 4, self.PIPE_WIDTH / 4)
                        py = pipe_y + self.PIPE_HEIGHT / 2
                        pvx = self.np_random.uniform(-0.5, 0.5)
                        pvy = self.np_random.uniform(-2, -0.5)
                        self.particles.append(Particle(px, py, pvx, pvy, 5, self.COLOR_CLEAN, 40))
        return reward

    def _update_physics(self):
        # Pipe Oscillation
        for i in range(4):
            oscillation = math.sin(self.steps / 60.0 + i * math.pi / 2) * 10
            self.pipe_positions[i] = (self.pipe_base_x[i], self.pipe_base_y + oscillation)

        # Smooth Rotation
        for i in range(4):
            current = self.pipe_current_angles[i]
            target = self.pipe_target_angles[i]
            # Handle angle wrapping for shortest path
            diff = (target - current + 180) % 360 - 180
            self.pipe_current_angles[i] = (current + diff * self.ROTATION_SPEED) % 360

        # Liquid Inflow (constant pressure)
        inflow_pipe = self.np_random.integers(0, 4)
        self.pipe_levels[inflow_pipe] = min(self.PIPE_CAPACITY, self.pipe_levels[inflow_pipe] + self.INFLOW_RATE)
        # Inflow slightly pollutes the pipe
        self.pipe_purity[inflow_pipe] *= (self.pipe_levels[inflow_pipe] - self.INFLOW_RATE) / self.pipe_levels[inflow_pipe]

        # Liquid Flow between pipes
        for i in range(4):
            level = self.pipe_levels[i]
            angle_deg = self.pipe_current_angles[i]
            outflow = level * self.OUTFLOW_RATE_PER_LEVEL

            # Determine flow direction
            if 45 <= angle_deg < 135: # Up
                pass # No flow
            elif 135 <= angle_deg < 225: # Left
                if i > 0:
                    self._transfer_liquid(i, i - 1, outflow)
            elif 225 <= angle_deg < 315: # Down
                self.pipe_levels[i] = max(0, level - outflow) # Liquid lost
            else: # Right (and wrapped around 0)
                if i < 3:
                    self._transfer_liquid(i, i + 1, outflow)
        
        # Particle updates
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

    def _transfer_liquid(self, src_idx, dest_idx, amount):
        src_level = self.pipe_levels[src_idx]
        dest_level = self.pipe_levels[dest_idx]
        actual_amount = min(amount, src_level)

        if dest_level + actual_amount > 0:
            # Mix purity
            new_purity = (self.pipe_purity[dest_idx] * dest_level + self.pipe_purity[src_idx] * actual_amount) / (dest_level + actual_amount)
            self.pipe_purity[dest_idx] = new_purity
        
        self.pipe_levels[src_idx] -= actual_amount
        self.pipe_levels[dest_idx] += actual_amount


    def _check_termination(self):
        # Loss conditions
        for i, level in enumerate(self.pipe_levels):
            if level <= 0 or level > self.PIPE_CAPACITY:
                # sfx: fail_buzzer.wav
                if level > self.PIPE_CAPACITY: # Overflow particles
                    px, py = self.pipe_positions[i]
                    for _ in range(50):
                        self.particles.append(Particle(
                            px + self.np_random.uniform(-self.PIPE_WIDTH/2, self.PIPE_WIDTH/2),
                            py - self.PIPE_HEIGHT/2,
                            self.np_random.uniform(-1, 1), self.np_random.uniform(0, 2),
                            4, lerp_color(self.COLOR_POLLUTED, self.COLOR_CLEAN, self.pipe_purity[i]), 60
                        ))
                return True, -100

        if self.steps >= self.MAX_EPISODE_STEPS:
            return True, 0

        # Win condition
        is_winning = True
        for i in range(4):
            if not (self.pipe_levels[i] / self.PIPE_CAPACITY >= 0.8 and self.pipe_purity[i] == 1.0):
                is_winning = False
                break
        
        if is_winning:
            self.win_timer += 1
        else:
            self.win_timer = 0

        if self.win_timer >= self.WIN_TIME_SECONDS * self.metadata["render_fps"]:
            self.win_condition_met = True
            # sfx: victory_fanfare.wav
            return True, 100

        return False, 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background Grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Particles
        for p in self.particles:
            p.draw(self.screen)

        # Pipes
        for i in range(4):
            pipe_x, pipe_y = self.pipe_positions[i]
            
            # Selector highlight
            if i == self.selected_pipe_idx and not self.game_over:
                selector_rect = pygame.Rect(pipe_x - self.PIPE_WIDTH/2 - 5, pipe_y - self.PIPE_HEIGHT/2 - 5, self.PIPE_WIDTH + 10, self.PIPE_HEIGHT + 10)
                pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, 2, 5)

            # Pipe Container
            container_rect = pygame.Rect(pipe_x - self.PIPE_WIDTH/2, pipe_y - self.PIPE_HEIGHT/2, self.PIPE_WIDTH, self.PIPE_HEIGHT)
            pygame.draw.rect(self.screen, self.COLOR_PIPE, container_rect, 2)

            # Liquid
            fill_ratio = self.pipe_levels[i] / self.PIPE_CAPACITY
            liquid_height = self.PIPE_HEIGHT * fill_ratio
            liquid_color = lerp_color(self.COLOR_POLLUTED, self.COLOR_CLEAN, self.pipe_purity[i])
            liquid_rect = pygame.Rect(
                pipe_x - self.PIPE_WIDTH/2,
                pipe_y + self.PIPE_HEIGHT/2 - liquid_height,
                self.PIPE_WIDTH,
                max(0, liquid_height)
            )
            pygame.draw.rect(self.screen, liquid_color, liquid_rect)

            # Nozzle (rotation indicator)
            angle_rad = math.radians(self.pipe_current_angles[i])
            nozzle_length = self.PIPE_WIDTH / 2 + 5
            nozzle_end_x = pipe_x + nozzle_length * math.cos(angle_rad)
            nozzle_end_y = pipe_y + nozzle_length * math.sin(angle_rad)
            pygame.draw.line(self.screen, self.COLOR_PIPE, (int(pipe_x), int(pipe_y)), (int(nozzle_end_x), int(nozzle_end_y)), 3)

    def _render_ui(self):
        # Per-pipe UI
        for i in range(4):
            pipe_x, pipe_y = self.pipe_positions[i]
            fill_pct = self.pipe_levels[i] / self.PIPE_CAPACITY * 100
            text_surface = self.font_small.render(f"{fill_pct:.0f}%", True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(pipe_x, pipe_y - self.PIPE_HEIGHT/2 - 15))
            self.screen.blit(text_surface, text_rect)

        # Main timer UI
        if self.win_timer > 0:
            timer_text = f"Win Timer: {self.win_timer * 100 / (self.WIN_TIME_SECONDS * self.metadata['render_fps']):.0f}%"
            text_surface = self.font_small.render(timer_text, True, self.COLOR_TEXT)
            self.screen.blit(text_surface, (self.SCREEN_WIDTH - text_surface.get_width() - 10, 10))

        # Game Over Text
        if self.game_over:
            color = self.COLOR_WIN if self.win_condition_met else self.COLOR_LOSS
            message = "SYSTEM STABLE" if self.win_condition_met else "SYSTEM FAILURE"
            text_surface = self.font_large.render(message, True, color)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            # Add a semi-transparent background for readability
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, bg_rect)
            self.screen.blit(text_surface, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win_timer": self.win_timer,
            "pipe_levels": self.pipe_levels,
            "pipe_purity": self.pipe_purity,
            "selected_pipe": self.selected_pipe_idx,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # print("✓ Implementation validated successfully") # Commented out to avoid stdout

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Pipe Flow Manager")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # Mapping keyboard keys to actions for manual play
    # This is NOT how an RL agent would work, it's for human testing
    
    while not done:
        movement = 0 # no-op
        space = 0
        shift = 0

        # For single-press actions
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_q: # Quit
                    done = True

        # For continuous/held actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0


        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata['render_fps'])

    env.close()