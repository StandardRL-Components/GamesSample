
# Generated: 2025-08-28T05:15:57.225768
# Source Brief: brief_02567.md
# Brief Index: 2567

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ←→ to run, ↑ to jump. Collect coins and reach the green flag before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced pixel art platformer. Navigate a procedurally generated world, collect coins for score, and race against the clock to reach the goal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_DEEP = (10, 5, 30)
    COLOR_BG_MID = (20, 15, 50)
    COLOR_BG_NEAR = (30, 25, 70)
    COLOR_PLATFORM = (139, 69, 19) # Brown
    COLOR_COIN = (255, 223, 0) # Gold
    COLOR_PLAYER = (0, 255, 255) # Bright Cyan
    COLOR_PLAYER_TRAIL = (0, 128, 128)
    COLOR_FLAG = (0, 255, 0) # Green
    COLOR_PIT_HAZARD = (255, 0, 0) # Red
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (0, 0, 0)

    # Physics
    FPS = 30
    GRAVITY = 0.8
    PLAYER_JUMP_STRENGTH = -15
    PLAYER_MOVE_SPEED = 6
    PLAYER_FRICTION = 0.85
    
    # Game settings
    MAX_STEPS = 60 * FPS # 60 seconds
    LEVEL_LENGTH_PIXELS = 10000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('monospace', 24, bold=True)
        self.font_game_over = pygame.font.SysFont('monospace', 60, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.on_ground = False
        
        self.platforms = []
        self.coins = []
        self.end_flag = None
        
        self.camera_x = 0
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        self.victory = False

        self.player_trail = []
        self.particles = []
        self.background_layers = []

        # Initialize state by calling reset
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def _generate_level(self):
        self.platforms = []
        self.coins = []
        
        # Starting platform
        last_platform_x = 0
        last_platform_y = self.SCREEN_HEIGHT - 50
        self.platforms.append(pygame.Rect(0, last_platform_y, 300, 50))
        
        # Procedurally generate platforms
        current_x = 300
        while current_x < self.LEVEL_LENGTH_PIXELS:
            gap = self.np_random.integers(80, 160)
            width = self.np_random.integers(100, 400)
            height_change = self.np_random.integers(-80, 80)
            
            current_x += gap
            y = np.clip(last_platform_y + height_change, 150, self.SCREEN_HEIGHT - 50)
            
            platform_rect = pygame.Rect(current_x, y, width, self.SCREEN_HEIGHT - y)
            self.platforms.append(platform_rect)

            # Add coins above the platform
            num_coins = self.np_random.integers(1, 5)
            for i in range(num_coins):
                coin_x = current_x + (width / (num_coins + 1)) * (i + 1)
                coin_y = y - 40
                self.coins.append(pygame.Rect(coin_x, coin_y, 10, 10))

            current_x += width
            last_platform_y = y

        # End flag on the last platform
        last_plat = self.platforms[-1]
        self.end_flag = pygame.Rect(last_plat.centerx, last_plat.top - 50, 20, 50)

    def _generate_background(self):
        self.background_layers = []
        for i in range(3): # 3 layers
            layer = []
            scroll_speed = 0.2 + i * 0.2
            for _ in range(50):
                x = self.np_random.uniform(-self.SCREEN_WIDTH, self.LEVEL_LENGTH_PIXELS * 1.5)
                y = self.np_random.uniform(0, self.SCREEN_HEIGHT)
                size = self.np_random.uniform(20, 100) * (i + 1)
                layer.append({'rect': pygame.Rect(x, y, size, size), 'speed': scroll_speed})
            self.background_layers.append(layer)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(100, self.SCREEN_HEIGHT - 100)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, 24, 32)
        self.on_ground = False
        
        self._generate_level()
        self._generate_background()
        
        self.camera_x = 0
        self.steps = 0
        self.score = 0
        self.timer = self.MAX_STEPS
        self.game_over = False
        self.victory = False

        self.player_trail = []
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.02  # Time penalty
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            
            # --- Player Input ---
            # Horizontal movement
            if movement == 3: # Left
                self.player_vel.x -= self.PLAYER_MOVE_SPEED * 0.2
            elif movement == 4: # Right
                self.player_vel.x += self.PLAYER_MOVE_SPEED * 0.2
                reward += 0.1 # Reward for moving forward

            # Jumping
            if movement == 1 and self.on_ground:
                self.player_vel.y = self.PLAYER_JUMP_STRENGTH
                self.on_ground = False
                # Sound: Jump
                # Particle effect for jump
                for _ in range(10):
                    self.particles.append(self._create_particle(self.player_rect.midbottom, self.COLOR_PLATFORM))

            # --- Physics & State Update ---
            # Apply gravity
            self.player_vel.y += self.GRAVITY
            
            # Apply friction
            if self.on_ground:
                self.player_vel.x *= self.PLAYER_FRICTION

            # Clamp velocity
            self.player_vel.x = np.clip(self.player_vel.x, -self.PLAYER_MOVE_SPEED, self.PLAYER_MOVE_SPEED)
            self.player_vel.y = np.clip(self.player_vel.y, -self.PLAYER_JUMP_STRENGTH * 2, 20)

            # Move player
            self.player_pos.x += self.player_vel.x
            self.player_pos.y += self.player_vel.y
            self.player_rect.center = self.player_pos

            # --- Collisions ---
            self.on_ground = False
            for plat in self.platforms:
                if self.player_rect.colliderect(plat) and self.player_vel.y >= 0:
                    # Check if player was above the platform in the previous frame
                    if self.player_pos.y - self.player_vel.y <= plat.top + self.player_rect.height / 2:
                        self.player_rect.bottom = plat.top
                        self.player_pos.y = self.player_rect.centery
                        if not self.on_ground: # First frame of landing
                            # Particle effect for landing
                            for _ in range(5):
                                self.particles.append(self._create_particle(self.player_rect.midbottom, self.COLOR_PLATFORM))
                        self.player_vel.y = 0
                        self.on_ground = True
                        break
            
            # Coin collection
            for coin in self.coins[:]:
                if self.player_rect.colliderect(coin):
                    self.coins.remove(coin)
                    self.score += 1
                    reward += 1.0
                    # Sound: Coin collect
                    # Particle effect for coin
                    for _ in range(15):
                        self.particles.append(self._create_particle(coin.center, self.COLOR_COIN))

            # --- Update Timers and Game State ---
            self.timer -= 1
            self.steps += 1
            
            # Update camera to follow player
            self.camera_x += (self.player_pos.x - self.camera_x - self.SCREEN_WIDTH / 3) * 0.1
            self.camera_x = max(0, self.camera_x)

            # Update player trail
            if self.steps % 3 == 0:
                self.player_trail.append(self.player_rect.copy())
                if len(self.player_trail) > 5:
                    self.player_trail.pop(0)

        # --- Termination Conditions ---
        terminated = False
        if self.player_rect.top > self.SCREEN_HEIGHT:
            self.game_over = True
            terminated = True
            reward -= 10.0 # Pitfall penalty
        
        if self.timer <= 0 and not self.victory:
            self.game_over = True
            terminated = True
        
        if self.end_flag and self.player_rect.colliderect(self.end_flag):
            self.game_over = True
            self.victory = True
            terminated = True
            reward += 100.0 # Victory reward
            self.score += 100
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _create_particle(self, pos, color):
        return {
            'pos': pygame.Vector2(pos),
            'vel': pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-4, 0)),
            'life': self.np_random.integers(10, 20),
            'color': color
        }

    def _render_game(self):
        # Render background layers with parallax
        for i, layer in enumerate(self.background_layers):
            color = [c - i * 10 for c in self.COLOR_BG_NEAR]
            color = [max(0, c) for c in color]
            for bg_obj in layer:
                obj_rect = bg_obj['rect']
                scroll_x = self.camera_x * bg_obj['speed']
                render_pos = (obj_rect.x - scroll_x, obj_rect.y)
                if -obj_rect.width < render_pos[0] < self.SCREEN_WIDTH:
                     pygame.draw.rect(self.screen, color, (*render_pos, obj_rect.width, obj_rect.height))

        # Render pitfall hazard indicator
        pit_gradient = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        for i in range(50):
            alpha = int(i / 50 * 150)
            pygame.draw.line(pit_gradient, (*self.COLOR_PIT_HAZARD, alpha), (0, i), (self.SCREEN_WIDTH, i))
        self.screen.blit(pit_gradient, (0, self.SCREEN_HEIGHT - 50))

        # Render platforms
        for plat in self.platforms:
            render_rect = plat.move(-self.camera_x, 0)
            if render_rect.right > 0 and render_rect.left < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, render_rect)

        # Render coins with spinning animation
        coin_width = 5 + 5 * math.sin(self.steps * 0.3)
        for coin in self.coins:
            render_rect = coin.move(-self.camera_x, 0)
            if render_rect.right > 0 and render_rect.left < self.SCREEN_WIDTH:
                display_coin = pygame.Rect(render_rect.centerx - coin_width/2, render_rect.y, coin_width, coin.height)
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, display_coin)
                pygame.gfxdraw.aaellipse(self.screen, int(display_coin.centerx), int(display_coin.centery), int(display_coin.width/2), int(display_coin.height/2), self.COLOR_COIN)

        # Render end flag
        if self.end_flag:
            render_rect = self.end_flag.move(-self.camera_x, 0)
            if render_rect.right > 0 and render_rect.left < self.SCREEN_WIDTH:
                pole_rect = pygame.Rect(render_rect.left, render_rect.top, 4, render_rect.height)
                flag_points = [
                    (render_rect.left + 4, render_rect.top),
                    (render_rect.left + 24, render_rect.top + 10),
                    (render_rect.left + 4, render_rect.top + 20)
                ]
                pygame.draw.rect(self.screen, (200, 200, 200), pole_rect)
                pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

        # Render particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'].y += 0.2 # Gravity on particles
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                render_pos = p['pos'] - pygame.Vector2(self.camera_x, 0)
                size = max(0, p['life'] * 0.3)
                pygame.draw.rect(self.screen, p['color'], (*render_pos, size, size))

        # Render player trail
        for i, trail_rect in enumerate(self.player_trail):
            alpha = (i + 1) / len(self.player_trail) * 100
            trail_surface = pygame.Surface(trail_rect.size, pygame.SRCALPHA)
            trail_surface.fill((*self.COLOR_PLAYER_TRAIL, alpha))
            render_pos = (trail_rect.x - self.camera_x, trail_rect.y)
            self.screen.blit(trail_surface, render_pos)

        # Render player with squash and stretch
        squash = max(0, 1 - abs(self.player_vel.y) * 0.02)
        stretch = max(0, self.player_vel.y * -0.01)
        display_height = self.player_rect.height * squash + stretch * 10
        display_width = self.player_rect.width / squash
        display_rect = pygame.Rect(
            self.player_rect.centerx - display_width / 2 - self.camera_x,
            self.player_rect.bottom - display_height,
            display_width,
            display_height
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, display_rect)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, (15, 10), self.font_ui)
        
        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        self._draw_text(timer_text, (self.SCREEN_WIDTH - 175, 10), self.font_ui)
        
        # Game Over / Victory message
        if self.game_over:
            if self.victory:
                msg = "VICTORY!"
                color = self.COLOR_FLAG
            else:
                msg = "GAME OVER"
                color = self.COLOR_PIT_HAZARD
            
            text_surf = self.font_game_over.render(msg, False, color)
            pos = (self.SCREEN_WIDTH/2 - text_surf.get_width()/2, self.SCREEN_HEIGHT/2 - text_surf.get_height()/2)
            self._draw_text(msg, pos, self.font_game_over, color=color)

    def _draw_text(self, text, pos, font, color=COLOR_UI_TEXT, shadow_color=COLOR_UI_SHADOW):
        shadow_surf = font.render(text, False, shadow_color)
        self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, False, color)
        self.screen.blit(text_surf, pos)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG_DEEP)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.timer / self.FPS),
            "player_x": self.player_pos.x,
            "player_y": self.player_pos.y,
            "victory": self.victory,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    # Set a dummy video driver to run headless
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv()
    
    # --- Manual Play Example ---
    # To play manually, you need a window. Comment out the os.environ line above
    # and run this section instead of the headless example.
    # env = GameEnv(render_mode="rgb_array")
    # screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    # pygame.display.set_caption("Pixel Platformer")
    # obs, info = env.reset()
    # done = False
    # while not done:
    #     action = [0, 0, 0] # Default no-op
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True

    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_LEFT]:
    #         action[0] = 3
    #     elif keys[pygame.K_RIGHT]:
    #         action[0] = 4
    #     if keys[pygame.K_UP]:
    #         action[0] = 1

    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated

    #     # Display the observation from the environment
    #     frame = np.transpose(obs, (1, 0, 2))
    #     surf = pygame.surfarray.make_surface(frame)
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()

    #     env.clock.tick(GameEnv.FPS)
    
    # env.close()


    # --- Headless RL Agent Example ---
    print("Running headless test...")
    obs, info = env.reset()
    total_reward = 0
    for i in range(2000):
        action = env.action_space.sample() # Random agent
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i+1) % 500 == 0:
            print(f"Step {i+1}, Info: {info}, Current Reward: {reward:.2f}")
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps. Final Info: {info}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
    env.close()
    print("Headless test finished.")