
# Generated: 2025-08-28T02:57:20.197174
# Source Brief: brief_01867.md
# Brief Index: 1867

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press space to jump over the red obstacles. Time your jumps to the beat for a higher score!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-view rhythm runner. Leap over obstacles in time with the music to build your combo and reach the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (40, 60, 80)
    COLOR_GROUND = (50, 50, 70)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (255, 150, 150)
    COLOR_BEAT_INDICATOR = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    # Physics & Gameplay
    FPS = 30
    GRAVITY = 0.8
    JUMP_STRENGTH = -15
    GROUND_Y = 320
    PLAYER_X = 100
    BPM = 120
    BEAT_PERIOD = int(FPS / (BPM / 60)) # 15 frames
    BEAT_WINDOW = 2 # frames on either side of the beat for "on-beat" bonus
    
    # Level
    LEVEL_LENGTH_STEPS = 1000
    INITIAL_OBSTACLE_SPEED = 5
    INITIAL_SPAWN_RATE = BEAT_PERIOD * 4 # Spawn every 4 beats

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Etc...
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel_y = 0
        self.on_ground = True
        self.obstacles = []
        self.prev_space_held = False
        self.combo = 0
        self.successful_jumps = 0
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.next_spawn_step = 0
        self.last_beat_pulse = 0
        self.rng = None
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(self.PLAYER_X, self.GROUND_Y)
        self.player_vel_y = 0
        self.on_ground = True
        self.obstacles = []
        self.prev_space_held = False
        self.combo = 0
        self.successful_jumps = 0
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.next_spawn_step = self.INITIAL_SPAWN_RATE
        self.last_beat_pulse = 0

        # Seed the random number generator
        self.rng = np.random.default_rng(seed)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right (unused)
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean (unused)
        
        jump_attempted = space_held and not self.prev_space_held
        
        if jump_attempted and self.on_ground:
            # sfx: jump_sound()
            self.player_vel_y = self.JUMP_STRENGTH
            self.on_ground = False
            
            # Check if jump was on beat
            beat_offset = self.steps % self.BEAT_PERIOD
            if not (beat_offset <= self.BEAT_WINDOW or beat_offset >= self.BEAT_PERIOD - self.BEAT_WINDOW):
                reward -= 0.2 # Off-beat penalty
                self.combo = 0 # Off-beat jump breaks combo
            else:
                self.last_beat_pulse = 1.0 # Visual feedback for on-beat jump

        self.prev_space_held = space_held
        
        # --- Game Logic Update ---
        # Survival reward
        reward += 0.1
        
        # Player physics
        if not self.on_ground:
            self.player_vel_y += self.GRAVITY
            self.player_pos.y += self.player_vel_y
        
        # Ground collision
        if self.player_pos.y >= self.GROUND_Y:
            self.player_pos.y = self.GROUND_Y
            self.player_vel_y = 0
            if not self.on_ground: # Just landed
                # sfx: land_sound()
                self.on_ground = True
                self.combo = 0 # Reset combo on landing

        player_rect = self._get_player_rect()

        # Obstacle update
        new_obstacles = []
        for obs in self.obstacles:
            obs['rect'].x -= self.obstacle_speed
            
            # Check for clearing an obstacle
            if not obs['cleared'] and obs['rect'].right < player_rect.left:
                obs['cleared'] = True
                clear_reward = 1 + self.combo
                reward += clear_reward
                self.combo += 1
                self.successful_jumps += 1
                # sfx: clear_obstacle_sound()
            
            # Check for collision
            if player_rect.colliderect(obs['rect']):
                self.game_over = True
                reward = -10
                # sfx: collision_sound()
            
            # Keep obstacle if it's still on screen
            if obs['rect'].right > 0:
                new_obstacles.append(obs)
        self.obstacles = new_obstacles

        # Obstacle spawning
        if self.steps >= self.next_spawn_step and not self.game_over:
            height = self.rng.integers(20, 50)
            self.obstacles.append({
                'rect': pygame.Rect(self.SCREEN_WIDTH, self.GROUND_Y - height, 25, height),
                'cleared': False
            })
            # Set next spawn time with some randomness
            self.next_spawn_step += self.INITIAL_SPAWN_RATE - self.rng.integers(0, self.BEAT_PERIOD)

        # Difficulty scaling
        if self.successful_jumps > 0 and self.successful_jumps % 20 == 0:
            self.obstacle_speed += 0.05
            self.successful_jumps += 1 # prevent multiple increases for same jump count

        # --- Termination Check ---
        self.steps += 1
        terminated = self._check_termination()
        if terminated and not self.game_over:
             reward += 100 # Victory reward
             # sfx: victory_sound()

        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.LEVEL_LENGTH_STEPS:
            return True
        return False
    
    def _get_observation(self):
        # Clear screen with background gradient
        for i in range(self.SCREEN_HEIGHT):
            interp = i / self.SCREEN_HEIGHT
            color = tuple(int(self.COLOR_BG_TOP[j] * (1 - interp) + self.COLOR_BG_BOTTOM[j] * interp) for j in range(3))
            pygame.draw.line(self.screen, color, (0, i), (self.SCREEN_WIDTH, i))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

        # Beat indicator
        beat_progress = (self.steps % self.BEAT_PERIOD) / self.BEAT_PERIOD
        pulse = (math.sin(beat_progress * 2 * math.pi - math.pi/2) + 1) / 2 # Smooth pulse 0-1
        pulse = 0.2 + pulse * 0.8 # Scale from 0.2 to 1.0
        
        # On-beat jump flash
        if self.last_beat_pulse > 0:
            pulse = self.last_beat_pulse
            self.last_beat_pulse -= 0.05 # Fade out
        
        indicator_radius = int(20 + 15 * pulse)
        indicator_alpha = int(50 + 150 * pulse)
        
        # Use gfxdraw for anti-aliased circle
        indicator_color = (*self.COLOR_BEAT_INDICATOR, indicator_alpha)
        pygame.gfxdraw.aacircle(self.screen, 50, 50, indicator_radius, indicator_color)
        pygame.gfxdraw.filled_circle(self.screen, 50, 50, indicator_radius, indicator_color)

        # Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])
            glow_rect = obs['rect'].inflate(6, 6)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_OBSTACLE_GLOW, 50), s.get_rect(), border_radius=3)
            self.screen.blit(s, glow_rect.topleft)

        # Player
        player_rect = self._get_player_rect()
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        glow_rect = player_rect.inflate(8, 8)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_PLAYER_GLOW, 80), s.get_rect(), border_radius=5)
        self.screen.blit(s, glow_rect.topleft)

    def _render_text(self, text, font, position, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW):
        shadow = font.render(text, True, shadow_color)
        self.screen.blit(shadow, (position[0] + 2, position[1] + 2))
        surface = font.render(text, True, color)
        self.screen.blit(surface, position)

    def _render_ui(self):
        # Combo counter
        if self.combo > 1:
            self._render_text(f"COMBO x{self.combo}", self.font_main, (self.SCREEN_WIDTH - 250, 20))
        
        # Progress bar
        progress = min(1.0, self.steps / self.LEVEL_LENGTH_STEPS)
        bar_width = 200
        bar_height = 15
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = 20
        pygame.draw.rect(self.screen, self.COLOR_TEXT_SHADOW, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (bar_x, bar_y, int(bar_width * progress), bar_height), border_radius=4)
        
        # Score
        self._render_text(f"Score: {int(self.score)}", self.font_small, (20, 20))

        # Game Over / Win message
        if self.game_over:
            self._render_text("GAME OVER", self.font_main, (self.SCREEN_WIDTH // 2 - 120, self.SCREEN_HEIGHT // 2 - 50))
            self._render_text(f"Final Score: {int(self.score)}", self.font_small, (self.SCREEN_WIDTH // 2 - 90, self.SCREEN_HEIGHT // 2))
        elif self.steps >= self.LEVEL_LENGTH_STEPS:
            self._render_text("LEVEL COMPLETE!", self.font_main, (self.SCREEN_WIDTH // 2 - 180, self.SCREEN_HEIGHT // 2 - 50))
            self._render_text(f"Final Score: {int(self.score)}", self.font_small, (self.SCREEN_WIDTH // 2 - 90, self.SCREEN_HEIGHT // 2))

    def _get_player_rect(self):
        size = 30
        return pygame.Rect(int(self.player_pos.x - size / 2), int(self.player_pos.y - size), size, size)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "combo": self.combo,
            "successful_jumps": self.successful_jumps - 1 if self.successful_jumps > 0 else 0
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import os
    # Set the video driver to a dummy driver for headless execution
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Test random agent survival to validate initial difficulty
    done = False
    total_reward = 0
    step_count = 0
    while not done:
        # A random agent will mostly press nothing, so let's give it a chance to jump
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

    print("\n--- Random Agent Test ---")
    print(f"Survived for {step_count} steps.")
    print(f"Final Score: {total_reward:.2f}")
    
    # Check if random agent survives at least 50 steps
    if step_count >= 50:
        print("✓ Initial difficulty is survivable (agent lasted >= 50 steps).")
    else:
        print(f"✗ Agent only survived {step_count} steps. Initial difficulty might be too high (this can be due to randomness).")

    env.close()