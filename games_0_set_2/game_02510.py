
# Generated: 2025-08-27T20:35:08.942815
# Source Brief: brief_02510.md
# Brief Index: 2510

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to aim. Press ↑ to jump. Hold SPACE while jumping for a higher jump. Press ↓ to fall faster."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced neon platformer. Hop between procedurally generated platforms, collect coins, and reach the goal before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Game
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_EPISODE_STEPS = 2000
    TIME_LIMIT_STEPS = 1000
    GOAL_X = SCREEN_WIDTH * 4 # The x-coordinate to reach for victory

    # Colors
    COLOR_BG_TOP = (10, 0, 30)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_OUTLINE = (200, 255, 200)
    COLOR_COIN = (255, 223, 0)
    COLOR_TEXT = (220, 220, 255)
    
    # Platform Colors
    PLATFORM_COLORS = {
        "safe": ((0, 100, 255), (150, 200, 255)), # Fill, Outline
        "risky": ((150, 50, 255), (220, 180, 255)),
        "bonus": ((255, 180, 0), (255, 230, 150)),
    }

    # Physics
    GRAVITY = 0.5
    JUMP_VELOCITY_LOW = -9
    JUMP_VELOCITY_HIGH = -12
    FAST_FALL_ACCEL = 0.4
    AIR_CONTROL_ACCEL = 0.4
    MAX_VX = 5
    FRICTION = -0.15

    # Player
    PLAYER_SIZE = (20, 30)

    # Platforms
    INITIAL_PLATFORM_GAP = 50
    MAX_PLATFORM_GAP = 200
    PLATFORM_GAP_INCREASE_RATE = 0.5
    PLATFORM_GAP_INCREASE_INTERVAL = 50
    PLATFORM_HEIGHT = 15
    PLATFORM_MIN_WIDTH = 60
    PLATFORM_MAX_WIDTH = 120

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        # Initialize state variables to be populated in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_facing_right = None
        self.on_ground = None
        self.player_rect = None
        self.squash_factor = None
        
        self.platforms = None
        self.coins = None
        self.particles = None
        
        self.camera_offset_x = 0
        self.current_platform_gap = None
        
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        self.game_won = False
        
        self.np_random = None

        # This will be called once gym.make() is done
        # self.reset()
        # self.validate_implementation() # For dev; comment out for final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_left = self.TIME_LIMIT_STEPS
        self.game_over = False
        self.game_won = False
        
        self.camera_offset_x = 0
        self.current_platform_gap = self.INITIAL_PLATFORM_GAP
        
        self.platforms = deque()
        self.coins = deque()
        self.particles = deque()

        # Initial platform
        start_platform = pygame.Rect(50, self.SCREEN_HEIGHT - 50, 150, self.PLATFORM_HEIGHT)
        self.platforms.append({"rect": start_platform, "type": "safe"})
        
        # Player state
        self.player_pos = pygame.Vector2(start_platform.centerx, start_platform.top - self.PLAYER_SIZE[1])
        self.player_vel = pygame.Vector2(0, 0)
        self.player_facing_right = True
        self.on_ground = True
        self.squash_factor = 1.0
        self.player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE[0], self.PLAYER_SIZE[1])

        # Procedurally generate initial world
        while self.platforms[-1]["rect"].right < self.GOAL_X + self.SCREEN_WIDTH:
            self._generate_next_platform()
            
        # Final platform
        final_platform_y = self.np_random.integers(self.SCREEN_HEIGHT - 200, self.SCREEN_HEIGHT - 50)
        final_platform = pygame.Rect(self.GOAL_X, final_platform_y, 200, self.PLATFORM_HEIGHT)
        self.platforms.append({"rect": final_platform, "type": "bonus"})

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(self.FPS)
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1 # Unused in this game
        
        reward = 0
        old_player_x = self.player_pos.x

        # --- Action Logic ---
        if self.on_ground:
            # Horizontal movement on ground (aiming)
            if movement == 3: # Left
                self.player_facing_right = False
            elif movement == 4: # Right
                self.player_facing_right = True
            
            # Jumping
            if movement == 1: # Up
                self.on_ground = False
                self.player_vel.y = self.JUMP_VELOCITY_HIGH if space_held else self.JUMP_VELOCITY_LOW
                self.player_vel.x = self.MAX_VX if self.player_facing_right else -self.MAX_VX
                self._create_particles(self.player_rect.midbottom, 10, self.COLOR_PLAYER_OUTLINE) # Jump dust
                # sfx: jump
        else: # In the air
            # Air control
            if movement == 3: # Left
                self.player_vel.x = max(-self.MAX_VX, self.player_vel.x - self.AIR_CONTROL_ACCEL)
            elif movement == 4: # Right
                self.player_vel.x = min(self.MAX_VX, self.player_vel.x + self.AIR_CONTROL_ACCEL)
            # Fast fall
            if movement == 2: # Down
                self.player_vel.y += self.FAST_FALL_ACCEL

        # --- Physics Update ---
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY
        else:
            # Friction on ground
            if abs(self.player_vel.x) > 0:
                self.player_vel.x += self.FRICTION * np.sign(self.player_vel.x)
                if abs(self.player_vel.x) < 0.2:
                    self.player_vel.x = 0

        self.player_pos += self.player_vel
        self.player_rect.topleft = self.player_pos

        # --- Collision Detection ---
        self.on_ground = False
        for p_data in self.platforms:
            platform = p_data["rect"]
            if self.player_rect.colliderect(platform) and self.player_vel.y >= 0:
                # Check if player was above the platform in the previous frame
                if self.player_pos.y + self.PLAYER_SIZE[1] - self.player_vel.y <= platform.top:
                    self.player_pos.y = platform.top - self.PLAYER_SIZE[1]
                    self.player_vel.y = 0
                    self.player_vel.x = 0 # Stop horizontal movement on landing
                    self.on_ground = True
                    self.squash_factor = 0.7 # For squash and stretch effect
                    
                    if p_data["type"] == "risky":
                        reward -= 0.2
                    # sfx: land
                    break
        
        # --- Coin Collection ---
        collected_coins = []
        for coin in self.coins:
            if self.player_rect.colliderect(coin["rect"]):
                collected_coins.append(coin)
                self.score += coin["value"]
                reward += coin["value"]
                self._create_particles(coin["rect"].center, 15, self.COLOR_COIN)
                # sfx: coin_collect
        self.coins = deque([c for c in self.coins if c not in collected_coins])

        # --- Update Game State ---
        self.steps += 1
        self.time_left -= 1
        
        # Update difficulty
        if self.steps > 0 and self.steps % self.PLATFORM_GAP_INCREASE_INTERVAL == 0:
            self.current_platform_gap = min(self.MAX_PLATFORM_GAP, self.current_platform_gap + self.PLATFORM_GAP_INCREASE_RATE)

        # Update camera
        target_camera_x = self.player_pos.x - self.SCREEN_WIDTH / 3
        self.camera_offset_x += (target_camera_x - self.camera_offset_x) * 0.1 # Smooth camera
        
        # Update squash & stretch effect
        if self.squash_factor < 1.0:
            self.squash_factor += 0.1
        self.squash_factor = min(1.0, self.squash_factor)
        
        # Update particles
        self._update_particles()
        
        # Manage off-screen platforms/coins
        self._cull_objects()

        # --- Reward Calculation ---
        # Horizontal progress reward
        dx = self.player_pos.x - old_player_x
        if dx > 0:
            reward += dx * 0.1
        else:
            reward += dx * 0.01 # Smaller penalty for moving left

        # --- Termination Check ---
        terminated = False
        if self.player_pos.y > self.SCREEN_HEIGHT: # Fell off screen
            self.game_over = True
            terminated = True
            # sfx: fall
        
        if self.time_left <= 0: # Time ran out
            self.game_over = True
            terminated = True
            # sfx: timeout
        
        if self.player_rect.right >= self.GOAL_X: # Reached goal
            self.game_over = True
            self.game_won = True
            terminated = True
            reward += 100
            # sfx: win
            
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_next_platform(self):
        last_platform = self.platforms[-1]["rect"]
        
        # Position
        gap = self.np_random.uniform(self.current_platform_gap * 0.8, self.current_platform_gap * 1.2)
        new_x = last_platform.right + gap
        
        max_y_up = last_platform.top - 100
        max_y_down = last_platform.top + 100
        new_y = self.np_random.uniform(max_y_up, max_y_down)
        new_y = np.clip(new_y, 100, self.SCREEN_HEIGHT - self.PLATFORM_HEIGHT - 20)

        # Size and type
        p_type_roll = self.np_random.random()
        if p_type_roll < 0.1: # 10% chance for bonus
            p_type = "bonus"
            width = self.np_random.integers(self.PLATFORM_MIN_WIDTH, self.PLATFORM_MAX_WIDTH)
        elif p_type_roll < 0.3: # 20% chance for risky
            p_type = "risky"
            width = self.np_random.integers(self.PLATFORM_MIN_WIDTH * 0.7, self.PLATFORM_MIN_WIDTH)
        else: # 70% chance for safe
            p_type = "safe"
            width = self.np_random.integers(self.PLATFORM_MIN_WIDTH, self.PLATFORM_MAX_WIDTH)
        
        new_platform = pygame.Rect(new_x, new_y, width, self.PLATFORM_HEIGHT)
        self.platforms.append({"rect": new_platform, "type": p_type})
        
        # Add coins
        if self.np_random.random() < 0.6: # 60% chance to have coins
            num_coins = self.np_random.integers(1, 4)
            coin_value = 5 if p_type == "bonus" else 1
            for i in range(num_coins):
                coin_x = new_platform.left + (new_platform.width / (num_coins + 1)) * (i + 1)
                coin_y = new_platform.top - 40
                self.coins.append({
                    "pos": pygame.Vector2(coin_x, coin_y),
                    "rect": pygame.Rect(coin_x - 5, coin_y - 5, 10, 10),
                    "value": coin_value,
                    "phase": self.np_random.uniform(0, 2 * math.pi)
                })

    def _cull_objects(self):
        cull_line = self.camera_offset_x - 100
        self.platforms = deque([p for p in self.platforms if p["rect"].right > cull_line])
        self.coins = deque([c for c in self.coins if c["pos"].x > cull_line])

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 0)),
                "lifespan": self.np_random.integers(10, 20),
                "color": color
            })
            
    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"].y += 0.2 # Particle gravity
            p["lifespan"] -= 1
        self.particles = deque([p for p in self.particles if p["lifespan"] > 0])

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background gradient
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # Render Particles
        for p in self.particles:
            screen_pos = (int(p["pos"].x - self.camera_offset_x), int(p["pos"].y))
            size = max(1, int(p["lifespan"] / 4))
            pygame.draw.circle(self.screen, p["color"], screen_pos, size)

        # Render Platforms
        for p_data in self.platforms:
            platform = p_data["rect"].copy()
            platform.x -= self.camera_offset_x
            fill_color, outline_color = self.PLATFORM_COLORS[p_data["type"]]
            pygame.draw.rect(self.screen, fill_color, platform, border_radius=3)
            pygame.draw.rect(self.screen, outline_color, platform, width=2, border_radius=3)

        # Render Coins
        for coin in self.coins:
            spin_factor = math.sin(self.steps * 0.2 + coin["phase"])
            width = int(5 + 4 * spin_factor)
            height = 10
            screen_pos = (coin["pos"].x - self.camera_offset_x, coin["pos"].y)
            
            if width > 0:
                coin_rect = pygame.Rect(screen_pos[0] - width/2, screen_pos[1] - height/2, width, height)
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, coin_rect)
                pygame.draw.ellipse(self.screen, (255,255,255), coin_rect, width=1)


        # Render Player
        width = int(self.PLAYER_SIZE[0] / self.squash_factor)
        height = int(self.PLAYER_SIZE[1] * self.squash_factor)
        
        player_screen_pos = (
            self.player_pos.x - self.camera_offset_x,
            self.player_pos.y + (self.PLAYER_SIZE[1] - height) # Adjust y to squash from bottom
        )
        
        player_draw_rect = pygame.Rect(player_screen_pos[0], player_screen_pos[1], width, height)
        player_draw_rect.centerx = int(self.player_pos.x - self.camera_offset_x)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_draw_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_draw_rect, width=2, border_radius=4)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Time
        time_text = self.font_large.render(f"TIME: {self.time_left}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)

        # Progress Bar
        progress = (self.player_pos.x / self.GOAL_X)
        bar_width = self.SCREEN_WIDTH - 40
        bar_height = 10
        
        bar_x = 20
        bar_y = self.SCREEN_HEIGHT - 30
        
        pygame.draw.rect(self.screen, (50, 50, 80), (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (bar_x, bar_y, int(bar_width * progress), bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, (bar_x, bar_y, bar_width, bar_height), width=1, border_radius=3)
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WON!" if self.game_won else "GAME OVER"
            msg_text = self.font_large.render(message, True, (255, 255, 255))
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(msg_text, msg_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "player_x": self.player_pos.x,
            "player_y": self.player_pos.y,
            "on_ground": self.on_ground,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
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
    # This block allows you to run the file directly to test the environment
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    pygame.display.set_caption("Neon Hop")
    game_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    # Main game loop
    running = True
    total_reward = 0
    while running:
        # Action mapping from keyboard to MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the game window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling (to close the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Final Info: {info}")
            total_reward = 0
            obs, info = env.reset()
            
    env.close()