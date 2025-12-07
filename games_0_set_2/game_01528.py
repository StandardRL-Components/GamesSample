
# Generated: 2025-08-27T17:25:18.746790
# Source Brief: brief_01528.md
# Brief Index: 1528

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press Space to jump to the beat. Time your jumps to avoid obstacles and build your combo."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-futuristic rhythm platformer. Navigate a vibrant, procedurally generated obstacle course by jumping to the beat."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_combo = pygame.font.Font(None, 48)
        self.font_combo_glow = pygame.font.Font(None, 52)
        
        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_GRID = (20, 40, 80)
        self.COLOR_PLAYER = (0, 255, 255) # Bright Cyan
        self.COLOR_PLAYER_GLOW = (0, 150, 150)
        self.COLOR_OBSTACLE = (255, 50, 50) # Red
        self.COLOR_OBSTACLE_GLOW = (150, 30, 30)
        self.COLOR_BEAT_INDICATOR = (255, 255, 255)
        self.COLOR_COMBO = (255, 215, 0) # Gold
        
        # Game constants
        self.MAX_STEPS = 5000
        self.GROUND_Y = self.HEIGHT - 50
        self.PLAYER_X = 120
        self.PLAYER_SIZE = 20
        self.GRAVITY = 0.8
        self.JUMP_VELOCITY = -15
        
        # Rhythm constants
        self.BPM = 120
        self.BEAT_DURATION_MS = 60000 / self.BPM
        self.BEAT_TOLERANCE_MS = 60 # 60ms window on either side of the beat

        # Initialize state variables
        self.reset()
        
        # Validate implementation
        # self.validate_implementation() # Uncomment for testing
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_y = self.GROUND_Y
        self.player_vy = 0
        
        self.obstacles = []
        self.cleared_obstacles = set()
        
        self.combo = 0
        self.combo_anim_timer = 0
        
        self.total_time_ms = 0
        self.beat_timer_ms = 0
        self.last_space_held = False
        self.just_jumped = False

        self.obstacle_spawn_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # --- Time and Beat Management ---
        # Advance frame, assuming 30fps for consistent physics
        dt_ms = self.clock.tick(30)
        self.total_time_ms += dt_ms
        self.beat_timer_ms = (self.beat_timer_ms + dt_ms) % self.BEAT_DURATION_MS
        
        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        jump_action = space_held and not self.last_space_held
        self.last_space_held = space_held
        
        # --- Player Physics ---
        self.player_vy += self.GRAVITY
        self.player_y += self.player_vy
        
        on_ground = self.player_y >= self.GROUND_Y
        if on_ground:
            self.player_y = self.GROUND_Y
            self.player_vy = 0
        
        if jump_action and on_ground:
            # sfx: player_jump.wav
            self.player_vy = self.JUMP_VELOCITY
            self.just_jumped = True
            
            # Check jump timing for reward
            is_on_beat = (self.beat_timer_ms < self.BEAT_TOLERANCE_MS or 
                          self.beat_timer_ms > self.BEAT_DURATION_MS - self.BEAT_TOLERANCE_MS)
            
            if is_on_beat:
                # sfx: good_jump.wav
                reward += 1
                self.combo += 1
                self.combo_anim_timer = 10 # Animate for 10 frames
            else:
                # sfx: bad_jump.wav
                reward -= 0.2
                self.combo = 0
        
        # --- Obstacle Management ---
        # Update speed and spawn rate based on progression
        current_obstacle_speed = 2.0 + 0.05 * (self.steps // 200)
        spawn_rate_hz = 1.0 + 0.05 * (self.steps // 500)
        current_spawn_interval_ms = 1000 / spawn_rate_hz
        
        # Spawn new obstacles
        self.obstacle_spawn_timer += dt_ms
        if self.obstacle_spawn_timer >= current_spawn_interval_ms:
            self.obstacle_spawn_timer = 0
            # Randomly choose between a low obstacle (needs jump) or high (needs ducking)
            obstacle_type = self.np_random.choice(['low', 'high'])
            if obstacle_type == 'low':
                o_height = self.np_random.integers(20, 41)
                o_y = self.GROUND_Y - o_height + 5 # +5 for slight ground overlap
            else: # high obstacle
                o_height = self.np_random.integers(40, 71)
                o_y = self.GROUND_Y - o_height - 60 # 60px gap to clear
            
            new_obstacle = pygame.Rect(self.WIDTH, o_y, 30, o_height)
            self.obstacles.append(new_obstacle)

        # Move and check obstacles
        player_rect = pygame.Rect(self.PLAYER_X - self.PLAYER_SIZE // 2, self.player_y - self.PLAYER_SIZE, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        for obstacle in self.obstacles[:]:
            obstacle.x -= current_obstacle_speed
            
            # Collision check
            if player_rect.colliderect(obstacle):
                # sfx: player_hit.wav
                self.game_over = True
                reward += -50
                break
            
            # Reward for clearing an obstacle
            if obstacle.right < self.PLAYER_X and id(obstacle) not in self.cleared_obstacles:
                # sfx: obstacle_clear.wav
                reward += 5
                self.score += 5 # Add directly to score for immediate feedback
                self.cleared_obstacles.add(id(obstacle))
                
            # Despawn off-screen obstacles
            if obstacle.right < 0:
                self.obstacles.remove(obstacle)
        
        # --- Update State ---
        self.steps += 1
        self.score += reward
        
        # --- Termination Conditions ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if not self.game_over and self.steps >= self.MAX_STEPS:
            # sfx: level_complete.wav
            reward += 100
            self.score += 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        
        # Beat pulsation calculation
        pulse_progress = self.beat_timer_ms / self.BEAT_DURATION_MS
        # Use a sharp attack-decay curve for a "thump" effect
        pulse_alpha = max(0, 1 - pulse_progress) ** 4
        
        # Pulsating Grid
        grid_color = self.COLOR_GRID
        pulse_color = tuple(min(255, c + int(pulse_alpha * 80)) for c in grid_color)
        
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, pulse_color, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, pulse_color, (0, i), (self.WIDTH, i), 1)
            
        # Ground line
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 2)
        
        # Beat Indicator
        indicator_size = int(10 + 30 * pulse_alpha)
        indicator_color = (255, 255, 255, int(255 * pulse_alpha))
        s = pygame.Surface((indicator_size * 2, indicator_size * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, indicator_size, indicator_size, indicator_size, indicator_color)
        self.screen.blit(s, (self.WIDTH // 2 - indicator_size, self.HEIGHT // 4 - indicator_size))

    def _render_player(self):
        player_rect = pygame.Rect(
            int(self.PLAYER_X - self.PLAYER_SIZE / 2),
            int(self.player_y - self.PLAYER_SIZE),
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        
        # Glow effect
        glow_size = int(self.PLAYER_SIZE * 1.5)
        s = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, glow_size, glow_size, glow_size, (*self.COLOR_PLAYER_GLOW, 80))
        self.screen.blit(s, (player_rect.centerx - glow_size, player_rect.centery - glow_size))
        
        # Player body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Jump particles
        if self.just_jumped:
            # Create some particles here if desired
            self.just_jumped = False

    def _render_obstacles(self):
        for obstacle in self.obstacles:
            # Glow effect
            glow_size = int(max(obstacle.width, obstacle.height) * 0.75)
            s = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, glow_size, glow_size, glow_size, (*self.COLOR_OBSTACLE_GLOW, 100))
            self.screen.blit(s, (obstacle.centerx - glow_size, obstacle.centery - glow_size))
            
            # Obstacle body
            pygame.gfxdraw.box(self.screen, obstacle, self.COLOR_OBSTACLE)
    
    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 20))
        
        # Combo display
        if self.combo > 1:
            if self.combo_anim_timer > 0:
                # Bouncing animation on combo increase
                scale = 1 + 0.3 * (self.combo_anim_timer / 10)
                self.combo_anim_timer -= 1
            else:
                scale = 1.0

            font_size = int(48 * scale)
            temp_font = pygame.font.Font(None, font_size)
            combo_surf = temp_font.render(f"x{self.combo}", True, self.COLOR_COMBO)
            
            # Glow for combo text
            glow_font_size = int(52 * scale)
            temp_glow_font = pygame.font.Font(None, glow_font_size)
            glow_surf = temp_glow_font.render(f"x{self.combo}", True, self.COLOR_COMBO)
            glow_surf.set_alpha(100)
            
            pos_x = self.PLAYER_X - combo_surf.get_width() // 2
            pos_y = self.player_y - self.PLAYER_SIZE - 60
            
            self.screen.blit(glow_surf, (pos_x - (glow_surf.get_width() - combo_surf.get_width()) // 2, pos_y - (glow_surf.get_height() - combo_surf.get_height()) // 2))
            self.screen.blit(combo_surf, (pos_x, pos_y))

    def _get_observation(self):
        self._render_background()
        self._render_obstacles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "combo": self.combo,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Player Controls ---
    # We map keyboard keys to the MultiDiscrete action space
    # Action: [movement, space, shift]
    key_to_action = {
        pygame.K_UP:    [1, 0, 0],
        pygame.K_DOWN:  [2, 0, 0],
        pygame.K_LEFT:  [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
    }
    
    # --- Game Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Re-initialize pygame for display
    pygame.display.set_caption("Rhythm Jumper")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    while not terminated:
        # Default action is no-op
        action = [0, 0, 0] 
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # Keyboard state for continuous actions
        keys = pygame.key.get_pressed()
        
        # Movement (unused in this game but kept for compatibility)
        for key, move_action in key_to_action.items():
            if keys[key]:
                action[0] = move_action[0]
                
        # Space and Shift
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
    env.close()