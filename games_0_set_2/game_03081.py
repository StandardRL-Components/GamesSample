
# Generated: 2025-08-28T06:58:27.182182
# Source Brief: brief_03081.md
# Brief Index: 3081

        
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


class Particle:
    """A simple particle class for visual effects."""
    def __init__(self, pos, vel, life, color, radius):
        self.pos = list(pos)
        self.vel = list(vel)
        self.life = life
        self.max_life = life
        self.color = color
        self.radius = radius

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1

    def draw(self, surface, camera_x):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            current_radius = int(self.radius * (self.life / self.max_life))
            if current_radius > 0:
                # Create a temporary surface for transparency
                temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, current_radius, current_radius, current_radius, self.color + (alpha,))
                surface.blit(temp_surf, (int(self.pos[0] - camera_x - current_radius), int(self.pos[1] - current_radius)))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Collect yellow coins and reach the green goal!"
    )

    game_description = (
        "A fast-paced, procedurally generated platformer. Jump between platforms, collect coins, "
        "and reach the end of the level before falling off-screen or running out of time."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.LEVEL_END_X = 5000
        self.MAX_STEPS = 1800 # 60 seconds at 30fps

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        # Colors
        self.COLOR_BG = (15, 23, 42)
        self.COLOR_PLAYER = (239, 68, 68)
        self.COLOR_PLATFORM = (226, 232, 240)
        self.COLOR_COIN = (250, 204, 21)
        self.COLOR_GOAL = (74, 222, 128)
        self.COLOR_TEXT = (241, 245, 249)
        self.COLOR_TIME_BAR = (5, 150, 105)

        # Game physics and parameters
        self.GRAVITY = 0.4
        self.JUMP_STRENGTH = -9.0
        self.PLAYER_MAX_SPEED_X = 5.0
        self.PLAYER_ACCEL = 0.5
        self.PLAYER_FRICTION = 0.85

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_size = None
        self.on_ground = None
        self.player_trail = None
        
        self.platforms = None
        self.coins = None
        self.particles = None
        self.parallax_stars = None
        
        self.camera_x = None
        self.last_platform_x = None
        
        self.steps = None
        self.score = None
        self.game_over = None
        self.termination_reason = ""

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""

        self.player_size = (20, 20)
        self.player_pos = np.array([100.0, 150.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_trail = []
        self.on_ground = False
        
        self.camera_x = 0
        self.last_platform_x = 0

        self.platforms = []
        self.coins = []
        self.particles = []
        self.parallax_stars = [
            (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT), self.np_random.integers(1, 4))
            for _ in range(100)
        ]

        # Create initial platforms
        self._add_platform(0, 250, 200)
        self._generate_world_chunk()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        
        # Store pre-step state for reward calculation
        prev_player_pos = self.player_pos.copy()

        # Update game logic
        self._update_player(movement)
        self._update_world_elements()

        # Calculate rewards
        reward = self._calculate_reward(prev_player_pos)

        # Update step counter and check for termination
        self.steps += 1
        terminated, term_reward, reason = self._check_termination()
        reward += term_reward
        
        if terminated:
            self.game_over = True
            self.termination_reason = reason

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel[0] += self.PLAYER_ACCEL
        else: # No-op or up/down
            self.player_vel[0] *= self.PLAYER_FRICTION
        
        self.player_vel[0] = np.clip(self.player_vel[0], -self.PLAYER_MAX_SPEED_X, self.PLAYER_MAX_SPEED_X)

        # Vertical movement (Jump)
        if movement == 1 and self.on_ground:
            self.player_vel[1] = self.JUMP_STRENGTH
            # sfx: jump

        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        
        # Update position
        self.player_pos += self.player_vel
        self.on_ground = False

        # Collision with platforms
        player_rect = pygame.Rect(self.player_pos, self.player_size)
        for platform in self.platforms:
            if player_rect.colliderect(platform):
                # Check if landing on top
                if self.player_vel[1] > 0 and player_rect.bottom < platform.top + self.player_vel[1] + 1:
                    self.player_pos[1] = platform.top - self.player_size[1]
                    self.player_vel[1] = 0
                    self.on_ground = True
                    break
    
    def _update_world_elements(self):
        # Update camera
        self.camera_x += (self.player_pos[0] - self.camera_x - self.SCREEN_WIDTH / 3) * 0.1

        # Update player trail
        self.player_trail.append(self.player_pos.copy())
        if len(self.player_trail) > 10:
            self.player_trail.pop(0)

        # Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

        # Collect coins
        player_rect = pygame.Rect(self.player_pos, self.player_size)
        collected_coins = []
        for coin in self.coins:
            if player_rect.colliderect(coin):
                collected_coins.append(coin)
                self.score += 10 # Corresponds to reward of +1
                # sfx: coin_collect
                for _ in range(15):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 4)
                    vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                    self.particles.append(Particle(coin.center, vel, 20, self.COLOR_COIN, 4))
        self.coins = [c for c in self.coins if c not in collected_coins]

        # Procedural generation
        if self.last_platform_x < self.camera_x + self.SCREEN_WIDTH + 200:
            self._generate_world_chunk()
        
        # Prune old objects
        self.platforms = [p for p in self.platforms if p.right > self.camera_x - 50]
        self.coins = [c for c in self.coins if c.right > self.camera_x - 50]

    def _calculate_reward(self, prev_player_pos):
        reward = 0
        # Reward for moving right
        dx = self.player_pos[0] - prev_player_pos[0]
        reward += dx * 0.1
        
        # Penalty for moving down
        dy = self.player_pos[1] - prev_player_pos[1]
        if dy > 0:
            reward -= dy * 0.1
        
        # Coin reward is handled in termination checks for simplicity with RL
        # For this implementation, coin collection adds to score, which is a proxy for reward.
        # Let's add an explicit reward for coins here.
        num_collected = (self.score / 10) - ((self._get_info()['score'] - (self.score - (len(self.coins) - len([c for c in self.coins if not pygame.Rect(self.player_pos, self.player_size).colliderect(c)])) * 10)) / 10)
        if len(self.particles) > 0 and self.particles[-1].life == 19: # A bit of a hack to detect new collection
            reward += 1.0

        return reward

    def _check_termination(self):
        # Fell off screen
        if self.player_pos[1] > self.SCREEN_HEIGHT + 50:
            return True, -50.0, "Fell Off-Screen"
        
        # Reached goal
        if self.player_pos[0] > self.LEVEL_END_X:
            return True, 100.0, "Level Complete!"

        # Time ran out
        if self.steps >= self.MAX_STEPS:
            return True, -10.0, "Time's Up!"

        return False, 0.0, ""

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_background()
        self._render_game_objects()
        self._render_player()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, z in self.parallax_stars:
            screen_x = (x - self.camera_x / (z * 2)) % self.SCREEN_WIDTH
            color = (
                self.COLOR_PLATFORM[0] // z,
                self.COLOR_PLATFORM[1] // z,
                self.COLOR_PLATFORM[2] // z,
            )
            pygame.draw.circle(self.screen, color, (int(screen_x), int(y)), 2 - z // 2)

    def _render_game_objects(self):
        # Platforms
        for p in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, (int(p.x - self.camera_x), int(p.y), p.width, p.height))

        # Coins
        for c in self.coins:
            pygame.gfxdraw.filled_circle(self.screen, int(c.centerx - self.camera_x), int(c.centery), int(c.width / 2), self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, int(c.centerx - self.camera_x), int(c.centery), int(c.width / 2), self.COLOR_COIN)

        # Goal
        goal_rect = pygame.Rect(self.LEVEL_END_X, 0, 20, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, (int(goal_rect.x - self.camera_x), goal_rect.y, goal_rect.width, goal_rect.height))

        # Particles
        for p in self.particles:
            p.draw(self.screen, self.camera_x)

    def _render_player(self):
        # Trail
        for i, pos in enumerate(self.player_trail):
            alpha = int(255 * (i / len(self.player_trail)) * 0.5)
            trail_color = self.COLOR_PLAYER + (alpha,)
            size = self.player_size[0] * (i / len(self.player_trail))
            temp_surf = pygame.Surface((int(size), int(size)), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, trail_color, temp_surf.get_rect())
            self.screen.blit(temp_surf, (int(pos[0] - self.camera_x - size/2 + self.player_size[0]/2), int(pos[1] - size/2 + self.player_size[1]/2)))
            
        # Player
        player_rect_on_screen = pygame.Rect(
            int(self.player_pos[0] - self.camera_x),
            int(self.player_pos[1]),
            self.player_size[0],
            self.player_size[1]
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect_on_screen)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time Bar
        time_ratio = 1.0 - (self.steps / self.MAX_STEPS)
        bar_width = (self.SCREEN_WIDTH - 20) * time_ratio
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (9, self.SCREEN_HEIGHT - 21, self.SCREEN_WIDTH - 18, 12), 1)
        pygame.draw.rect(self.screen, self.COLOR_TIME_BAR, (10, self.SCREEN_HEIGHT - 20, max(0, bar_width), 10))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text_surface = self.font_game_over.render(self.termination_reason, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos.tolist(),
            "player_vel": self.player_vel.tolist(),
        }

    def _add_platform(self, x, y, width):
        self.platforms.append(pygame.Rect(x, y, width, 20))
        self.last_platform_x = max(self.last_platform_x, x + width)

    def _generate_world_chunk(self):
        # Difficulty scaling
        min_gap = 50 + self.steps * 0.05
        max_gap = 100 + self.steps * 0.05
        min_height_diff = -80
        max_height_diff = 80

        last_p = self.platforms[-1]
        
        for _ in range(10): # Generate 10 new platforms at a time
            if self.last_platform_x > self.LEVEL_END_X - 500: # Stop generating near the end
                break
                
            gap = self.np_random.uniform(min_gap, max_gap)
            height_diff = self.np_random.uniform(min_height_diff, max_height_diff)
            width = self.np_random.integers(80, 200)
            
            new_x = last_p.right + gap
            new_y = np.clip(last_p.y + height_diff, 100, self.SCREEN_HEIGHT - 50)
            
            self._add_platform(new_x, new_y, width)
            
            # Maybe add coins
            if self.np_random.random() < 0.6:
                num_coins = self.np_random.integers(1, 4)
                for i in range(num_coins):
                    coin_x = new_x + (width / (num_coins + 1)) * (i + 1)
                    coin_y = new_y - 40
                    self.coins.append(pygame.Rect(coin_x, coin_y, 14, 14))
            
            last_p = self.platforms[-1]

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Override screen for display
    env.screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Procedural Platformer")

    done = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        if keys[pygame.K_UP]:
            action[0] = 1
        
        # For manual play, pressing 'r' will reset the environment
        if keys[pygame.K_r]:
            done = True

        if not env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            total_reward = 0
            obs, info = env.reset()
            # Add a small delay before restarting
            pygame.time.wait(2000)
            
        env.clock.tick(30) # Run at 30 FPS

    pygame.quit()