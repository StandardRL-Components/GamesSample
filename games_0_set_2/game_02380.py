
# Generated: 2025-08-27T20:11:59.225817
# Source Brief: brief_02380.md
# Brief Index: 2380

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press space to squash bugs."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Squash waves of procedurally generated bugs before they reach the bottom of the screen."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.game_font = pygame.font.SysFont("monospace", 24)
        self.game_over_font = pygame.font.SysFont("monospace", 72, bold=True)
        
        # Colors
        self.COLOR_BG = (34, 139, 34)  # ForestGreen
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_OUTLINE = (0, 0, 0)
        self.COLOR_UI = (255, 255, 255)
        self.BUG_COLORS = [(255, 0, 0), (0, 0, 255), (255, 255, 0)] # Red, Blue, Yellow

        # Game constants
        self.MAX_STEPS = 3600  # 60 seconds at 60 FPS
        self.CURSOR_SPEED = 5
        self.WAVE_INTERVAL = 600 # 10 seconds at 60 FPS
        self.INITIAL_BUG_SPEED = 1.0
        self.BUG_SPEED_INCREMENT = 0.05
        
        # Initialize state variables
        self.cursor_pos = None
        self.bugs = None
        self.splats = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.won_game = None
        self.wave_timer = None
        self.current_bug_speed = None
        self.np_random = None
        
        self.reset()
        
        # This will run once and confirm the implementation is valid.
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.bugs = []
        self.splats = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.won_game = False
        
        self.wave_timer = 0
        self.current_bug_speed = self.INITIAL_BUG_SPEED
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        reward = 0.0
        
        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        # --- Handle Input ---
        self._handle_input(movement)
        
        if space_pressed:
            # sfx: squash.wav
            if self._handle_squash():
                reward += 1.0
                self.score += 10

        # --- Update Game State ---
        self.steps += 1
        self.wave_timer += 1
        
        self._update_splats()
        bug_reached_bottom = self._update_bugs()
        
        if self.wave_timer >= self.WAVE_INTERVAL:
            # sfx: new_wave.wav
            self.wave_timer = 0
            self.current_bug_speed += self.BUG_SPEED_INCREMENT
            self._spawn_wave()
            
        # --- Check Termination Conditions ---
        if bug_reached_bottom:
            # sfx: game_over.wav
            self.game_over = True
            reward = -100.0
        elif self.steps >= self.MAX_STEPS:
            # sfx: win.wav
            self.game_over = True
            self.won_game = True
            reward = 100.0
        else:
            # Survival reward
            reward += 0.1

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2:  # Down
            self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3:  # Left
            self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4:  # Right
            self.cursor_pos[0] += self.CURSOR_SPEED
        
        # Clamp cursor to screen bounds
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

    def _handle_squash(self):
        squashed = False
        for i in range(len(self.bugs) - 1, -1, -1):
            bug = self.bugs[i]
            dist = np.linalg.norm(self.cursor_pos - bug['pos'])
            if dist < bug['radius']:
                self._create_splat(bug['pos'], bug['color'], bug['radius'])
                self.bugs.pop(i)
                squashed = True
                break # Squash one bug per action
        return squashed

    def _create_splat(self, pos, color, radius):
        num_particles = 20
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            lifespan = self.np_random.integers(20, 40)
            particle_radius = self.np_random.uniform(1, radius / 3)
            
            self.splats.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color,
                'radius': particle_radius
            })

    def _update_bugs(self):
        for bug in self.bugs:
            bug['pos'][1] += bug['speed']
            if bug['pos'][1] - bug['radius'] > self.HEIGHT:
                return True # A bug reached the bottom
        return False

    def _update_splats(self):
        for i in range(len(self.splats) - 1, -1, -1):
            splat = self.splats[i]
            splat['pos'] += splat['vel']
            splat['lifespan'] -= 1
            splat['vel'] *= 0.95 # friction
            if splat['lifespan'] <= 0:
                self.splats.pop(i)

    def _spawn_wave(self):
        num_bugs = 3 + (self.steps // self.WAVE_INTERVAL)
        for _ in range(num_bugs):
            self.bugs.append({
                'pos': np.array([
                    self.np_random.integers(20, self.WIDTH - 20),
                    self.np_random.integers(-100, -20)
                ], dtype=np.float32),
                'speed': self.current_bug_speed + self.np_random.uniform(-0.2, 0.2),
                'radius': self.np_random.integers(8, 15),
                'color': random.choice(self.BUG_COLORS)
            })
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render splats
        for splat in self.splats:
            pos = (int(splat['pos'][0]), int(splat['pos'][1]))
            alpha = int(255 * (splat['lifespan'] / splat['max_lifespan']))
            if alpha > 0:
                # Pygame does not handle semi-transparent filled circles well without a separate surface
                # This is a common workaround.
                temp_surf = pygame.Surface((splat['radius']*2, splat['radius']*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, int(splat['radius']), int(splat['radius']), int(splat['radius']), (*splat['color'], alpha))
                self.screen.blit(temp_surf, (pos[0] - int(splat['radius']), pos[1] - int(splat['radius'])))

        # Render bugs
        for bug in self.bugs:
            pos = (int(bug['pos'][0]), int(bug['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], bug['radius'], bug['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], bug['radius'], (0,0,0))

        # Render cursor
        cursor_x, cursor_y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        size = 12
        # Outline
        pygame.draw.line(self.screen, self.COLOR_CURSOR_OUTLINE, (cursor_x - size - 1, cursor_y), (cursor_x + size + 1, cursor_y), 5)
        pygame.draw.line(self.screen, self.COLOR_CURSOR_OUTLINE, (cursor_x, cursor_y - size - 1), (cursor_x, cursor_y + size + 1), 5)
        # Crosshair
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_x - size, cursor_y), (cursor_x + size, cursor_y), 3)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y - size), (cursor_x, cursor_y + size), 3)

    def _render_ui(self):
        # Score
        score_text = self.game_font.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / 60)
        timer_text = self.game_font.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            if self.won_game:
                msg = "YOU WIN!"
                color = (255, 255, 0) # Gold
            else:
                msg = "GAME OVER"
                color = (255, 0, 0) # Red
            
            end_text = self.game_over_font.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            # Draw a shadow/outline for better readability
            shadow_text = self.game_over_font.render(msg, True, (0,0,0))
            self.screen.blit(shadow_text, (text_rect.x + 3, text_rect.y + 3))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }
        
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to run pygame in a window
    import os
    os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bug Squasher")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # --- Human Input ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0 # Not used in this game
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Render to screen ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Control FPS ---
        clock.tick(60)
        
    print(f"Game Over! Final Score: {info['score']}")
    pygame.quit()