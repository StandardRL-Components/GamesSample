import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = "Jump ever higher in this vertical platformer. Land on moving platforms to score points and reach the top."
    user_guide = "Press the space bar to jump. Land on platforms to ascend and score points."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (20, 40, 80)
    COLOR_PLAYER = (255, 80, 80)
    COLOR_PLAYER_GLOW = (255, 120, 120)
    COLOR_PLATFORM = (120, 200, 255)
    COLOR_PLATFORM_EDGE = (180, 230, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    # Physics
    GRAVITY = 0.8
    JUMP_VELOCITY = -15
    PLAYER_SIZE = 20
    PLATFORM_SPEED = 5 / FPS * 30  # 5 units/second
    PLATFORM_OSC_PERIOD = 2 * FPS # 2 seconds

    # Gameplay
    WIN_SCORE = 15
    MAX_EPISODE_STEPS = 1500 # Increased to allow for skilled play
    SCROLL_THRESHOLD = HEIGHT / 2.5

    # Rewards
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0
    REWARD_LAND = 1.0
    REWARD_ALIVE = 0.01 # Changed from 0.1 to keep total reward scale reasonable

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.background_surf = self._create_gradient_background()

        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.platforms = []
        self.on_platform = False
        self.last_space_held = False
        self.highest_landed_platform_idx = -1
        self.current_osc_amplitude = 0
        
        # self.reset() # Not strictly necessary, but can be useful for initialization
        # self.validate_implementation() # For development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player = {
            'rect': pygame.Rect(self.WIDTH / 2 - self.PLAYER_SIZE / 2, self.HEIGHT - 100, self.PLAYER_SIZE, self.PLAYER_SIZE),
            'vy': 0
        }
        self.on_platform = True
        self.last_space_held = False
        self.highest_landed_platform_idx = 0

        self._generate_initial_platforms()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, self.game_over, self.steps >= self.MAX_EPISODE_STEPS, self._get_info()

        reward = self.REWARD_ALIVE
        
        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        jump_requested = space_held and not self.last_space_held
        self.last_space_held = space_held
        
        # --- Player Update ---
        if jump_requested and self.on_platform:
            # // Play jump sound
            self.player['vy'] = self.JUMP_VELOCITY
            self.on_platform = False

        self.player['vy'] += self.GRAVITY
        self.player['rect'].y += self.player['vy']
        
        # --- Platform & Collision Update ---
        self.on_platform = False
        self._update_platforms()
        
        landing_reward = self._handle_collisions()
        reward += landing_reward

        # --- World Scrolling & Platform Management ---
        self._handle_scrolling()
        self._manage_platforms()

        self.steps += 1
        
        terminated = False
        truncated = False

        if self.score >= self.WIN_SCORE:
            terminated = True
            reward = self.REWARD_WIN
        elif self.player['rect'].top > self.HEIGHT:
            terminated = True
            reward = self.REWARD_LOSE
        
        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
        
        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_initial_platforms(self):
        self.platforms = []
        # Starting platform
        start_plat = {
            'rect': pygame.Rect(self.player['rect'].centerx - 50, self.player['rect'].bottom, 100, 20),
            'id': 0,
            'phase': self.np_random.uniform(0, 2 * math.pi),
            'amplitude': 0, # Start platform is static
            'speed_mult': 0 # Start platform is static horizontally
        }
        self.platforms.append(start_plat)
        
        # Procedurally generate next few platforms
        for i in range(1, 10):
            self._generate_new_platform()

    def _generate_new_platform(self):
        last_plat = self.platforms[-1]
        new_id = last_plat['id'] + 1

        difficulty_tier = self.score // 5
        self.current_osc_amplitude = 50 + difficulty_tier * 5

        # Horizontal position relative to last platform
        min_x_offset = -200
        max_x_offset = 200
        
        # Vertical distance based on jump height
        max_jump_height = abs((self.JUMP_VELOCITY**2) / (2 * self.GRAVITY))
        min_y_dist = max_jump_height * 0.4
        max_y_dist = max_jump_height * 0.85
        
        new_x = last_plat['rect'].x + self.np_random.integers(min_x_offset, max_x_offset)
        new_y = last_plat['rect'].y - self.np_random.integers(min_y_dist, max_y_dist)
        new_width = self.np_random.integers(80, 150)

        # Ensure platform is not generated off-screen
        new_x = np.clip(new_x, 0, self.WIDTH - new_width)

        new_plat = {
            'rect': pygame.Rect(new_x, new_y, new_width, 20),
            'id': new_id,
            'phase': self.np_random.uniform(0, 2 * math.pi),
            'amplitude': self.current_osc_amplitude,
            'speed_mult': 1 if self.np_random.random() > 0.5 else -1
        }
        self.platforms.append(new_plat)

    def _update_platforms(self):
        for plat in self.platforms:
            # Horizontal movement
            plat['rect'].x += self.PLATFORM_SPEED * plat['speed_mult']
            if plat['rect'].right < 0:
                plat['rect'].left = self.WIDTH
            elif plat['rect'].left > self.WIDTH:
                plat['rect'].right = 0
            
            # Vertical oscillation
            if 'base_y' not in plat:
                plat['base_y'] = plat['rect'].y
            oscillation = plat['amplitude'] * math.sin(plat['phase'] + (2 * math.pi * self.steps / self.PLATFORM_OSC_PERIOD))
            plat['rect'].y = plat['base_y'] + oscillation

    def _handle_collisions(self):
        reward = 0
        player_rect = self.player['rect']
        
        for plat in self.platforms:
            plat_rect = plat['rect']
            # Check for landing: player is falling, was above, and is now intersecting
            is_falling = self.player['vy'] > 0
            was_above = player_rect.bottom - self.player['vy'] <= plat_rect.top + 1 # Add tolerance
            
            if is_falling and was_above and player_rect.colliderect(plat_rect):
                # Side collisions count as landing
                if player_rect.left < plat_rect.right and player_rect.right > plat_rect.left:
                    self.player['rect'].bottom = plat_rect.top
                    self.player['vy'] = 0
                    self.on_platform = True
                    # // Play land sound
                    
                    if plat['id'] > self.highest_landed_platform_idx:
                        self.highest_landed_platform_idx = plat['id']
                        self.score += 1
                        reward = self.REWARD_LAND
                        # Generate new platforms as we progress
                        self._generate_new_platform()
                    break # Only land on one platform
        return reward
        
    def _handle_scrolling(self):
        if self.player['rect'].y < self.SCROLL_THRESHOLD:
            scroll_amount = self.SCROLL_THRESHOLD - self.player['rect'].y
            self.player['rect'].y += scroll_amount
            for plat in self.platforms:
                plat['rect'].y += scroll_amount
                if 'base_y' in plat:
                    plat['base_y'] += scroll_amount

    def _manage_platforms(self):
        # Remove platforms that are far below the screen
        self.platforms = [p for p in self.platforms if p['rect'].top < self.HEIGHT + 100]

    def _get_observation(self):
        self.screen.blit(self.background_surf, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_y": self.player['rect'].y,
            "player_vy": self.player['vy'],
        }

    def _render_game(self):
        # Render platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_EDGE, plat['rect'])
            inner_rect = plat['rect'].inflate(-4, -4)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, inner_rect)

        # Render player with glow
        player_center = self.player['rect'].center
        glow_radius = int(self.PLAYER_SIZE * 1.2)
        
        # Simple radial glow
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (player_center[0] - glow_radius, player_center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player['rect'])

    def _render_ui(self):
        score_text = f"PLATFORMS: {self.score}/{self.WIN_SCORE}"
        self._draw_text(score_text, (15, 15))

    def _draw_text(self, text, pos):
        shadow = self.font.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
        main = self.font.render(text, True, self.COLOR_TEXT)
        self.screen.blit(main, pos)

    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            # Linear interpolation between top and bottom colors
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set a non-dummy driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 

    env = GameEnv()
    # env.validate_implementation() # Optional validation
    
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Ascend Platformer")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no action

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        # Map keys to MultiDiscrete action
        if keys[pygame.K_SPACE]:
            action[1] = 1
        # Movement and Shift are not used in this game, but mapping is kept for completeness
        # if keys[pygame.K_UP]: action[0] = 1
        # if keys[pygame.K_DOWN]: action[0] = 2
        # if keys[pygame.K_LEFT]: action[0] = 3
        # if keys[pygame.K_RIGHT]: action[0] = 4
        # if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
            action.fill(0)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
    env.close()