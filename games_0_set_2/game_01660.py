import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set a dummy video driver for headless operation
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = "Controls: Use arrow keys (↑↓←→) to move your robot."

    # User-facing game description
    game_description = (
        "Control a robot in a top-down arena. Collect 50 coins before the 60-second timer runs out, "
        "but be careful to avoid the moving red obstacles!"
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = pygame.Color("#2c3e50")  # Dark Slate Blue
    COLOR_ARENA = pygame.Color("#34495e") # Slightly lighter Slate Blue
    COLOR_PLAYER = pygame.Color("#3498db") # Bright Blue
    COLOR_PLAYER_GLOW = pygame.Color("#5dade2")
    COLOR_COIN = pygame.Color("#f1c40f") # Yellow
    COLOR_OBSTACLE = pygame.Color("#e74c3c") # Red
    COLOR_TEXT = pygame.Color("#ecf0f1") # Off-white
    COLOR_TEXT_SHADOW = pygame.Color("#222222")

    # Game parameters
    PLAYER_SIZE = 20
    PLAYER_SPEED = 4
    COIN_SIZE = 8
    OBSTACLE_SIZE = 25
    WIN_CONDITION_COINS = 50
    NUM_INITIAL_COINS = 10
    NUM_OBSTACLES = 8
    INITIAL_OBSTACLE_SPEED = 1.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.player_rect = None
        self.coins = []
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.coins_collected_count = 0
        self.time_remaining = 0
        self.obstacle_speed_multiplier = 1.0
        self.game_over = False
        self.last_dist_to_coin = 0

        # Run validation
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.player_rect = pygame.Rect(
            self.SCREEN_WIDTH // 2 - self.PLAYER_SIZE // 2,
            self.SCREEN_HEIGHT // 2 - self.PLAYER_SIZE // 2,
            self.PLAYER_SIZE,
            self.PLAYER_SIZE,
        )
        self.steps = 0
        self.score = 0
        self.coins_collected_count = 0
        self.time_remaining = self.MAX_STEPS
        self.obstacle_speed_multiplier = 1.0
        self.game_over = False
        self.particles = []

        # Initialize coins and obstacles
        self.coins = [self._spawn_coin() for _ in range(self.NUM_INITIAL_COINS)]
        self.obstacles = [self._spawn_obstacle() for _ in range(self.NUM_OBSTACLES)]

        # Calculate initial distance to nearest coin for reward calculation
        self.last_dist_to_coin = self._get_dist_to_nearest_coin()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # Return a final observation but don't advance the state
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        # Unpack action
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        # --- Update Game Logic ---
        self.steps += 1
        self.time_remaining -= 1

        self._update_player(movement)
        self._update_obstacles()
        self._update_particles()

        # --- Handle Collisions and Events ---
        reward, coin_collected = self._handle_collisions()

        # --- Calculate Reward ---
        current_dist_to_coin = self._get_dist_to_nearest_coin()
        if not self.coins: # No coins left
            dist_reward = 0
        elif current_dist_to_coin < self.last_dist_to_coin:
            dist_reward = 0.1  # Moved closer
        else:
            dist_reward = -0.01 # Moved away or stayed same
        self.last_dist_to_coin = current_dist_to_coin
        reward += dist_reward

        # --- Check Termination ---
        terminated = False
        if self.coins_collected_count >= self.WIN_CONDITION_COINS:
            reward += 100  # Win bonus
            terminated = True
            self.game_over = True
        elif self.time_remaining <= 0:
            reward += -10  # Time out penalty
            terminated = True
            self.game_over = True
        elif self.game_over: # Set by obstacle collision
            reward += -50 # Obstacle collision penalty
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info(),
        )

    # --- Helper methods for game logic ---

    def _update_player(self, movement):
        if movement == 1:  # Up
            self.player_rect.y -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_rect.y += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_rect.x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_rect.x += self.PLAYER_SPEED

        # Clamp player position to screen bounds
        self.player_rect.left = max(0, self.player_rect.left)
        self.player_rect.right = min(self.SCREEN_WIDTH, self.player_rect.right)
        self.player_rect.top = max(0, self.player_rect.top)
        self.player_rect.bottom = min(self.SCREEN_HEIGHT, self.player_rect.bottom)

    def _update_obstacles(self):
        speed = self.INITIAL_OBSTACLE_SPEED * self.obstacle_speed_multiplier
        for obs in self.obstacles:
            obs['rect'].x += obs['vel'][0] * speed
            obs['rect'].y += obs['vel'][1] * speed

            if obs['rect'].left < 0 or obs['rect'].right > self.SCREEN_WIDTH:
                obs['vel'] = (-obs['vel'][0], obs['vel'][1])
                obs['rect'].left = max(0, obs['rect'].left)
                obs['rect'].right = min(self.SCREEN_WIDTH, obs['rect'].right)
            if obs['rect'].top < 0 or obs['rect'].bottom > self.SCREEN_HEIGHT:
                obs['vel'] = (obs['vel'][0], -obs['vel'][1])
                obs['rect'].top = max(0, obs['rect'].top)
                obs['rect'].bottom = min(self.SCREEN_HEIGHT, obs['rect'].bottom)


    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['radius'] -= 0.2
            p['lifespan'] -= 1

    def _handle_collisions(self):
        reward = 0
        coin_collected_this_step = False

        # Player vs Coins
        collided_coin_index = self.player_rect.collidelist(self.coins)
        if collided_coin_index != -1:
            collected_coin_pos = self.coins[collided_coin_index].center
            self._create_particles(collected_coin_pos, self.COLOR_COIN)

            self.coins.pop(collided_coin_index)
            self.coins.append(self._spawn_coin())

            reward += 10
            self.score += 10
            self.coins_collected_count += 1
            coin_collected_this_step = True

            # Difficulty scaling
            if self.coins_collected_count > 0 and self.coins_collected_count % 10 == 0:
                self.obstacle_speed_multiplier += 0.1

        # Player vs Obstacles
        for obs in self.obstacles:
            if self.player_rect.colliderect(obs['rect']):
                self.game_over = True
                self._create_particles(self.player_rect.center, self.COLOR_OBSTACLE, 20)
                break
        
        return reward, coin_collected_this_step

    def _get_dist_to_nearest_coin(self):
        if not self.coins:
            return 0
        player_pos = np.array(self.player_rect.center)
        coin_positions = np.array([c.center for c in self.coins])
        distances = np.linalg.norm(coin_positions - player_pos, axis=1)
        return np.min(distances)

    def _spawn_coin(self):
        while True:
            x = self.np_random.integers(self.COIN_SIZE, self.SCREEN_WIDTH - self.COIN_SIZE)
            y = self.np_random.integers(self.COIN_SIZE, self.SCREEN_HEIGHT - self.COIN_SIZE)
            new_coin = pygame.Rect(x, y, self.COIN_SIZE * 2, self.COIN_SIZE * 2)
            # Ensure it doesn't spawn inside an obstacle
            if new_coin.collidelist([o['rect'] for o in self.obstacles]) == -1:
                return new_coin

    def _spawn_obstacle(self):
        x = self.np_random.integers(0, self.SCREEN_WIDTH - self.OBSTACLE_SIZE)
        y = self.np_random.integers(0, self.SCREEN_HEIGHT - self.OBSTACLE_SIZE)
        rect = pygame.Rect(x, y, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE)
        
        angle = self.np_random.random() * 2 * math.pi
        vel = (math.cos(angle), math.sin(angle))
        
        return {'rect': rect, 'vel': vel}

    def _create_particles(self, pos, color, count=10):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos,
                'vel': vel,
                'radius': self.np_random.integers(4, 8),
                'lifespan': self.np_random.integers(10, 20),
                'color': color
            })

    # --- Rendering and Info ---

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), border_radius=10)

        # Render game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'], border_radius=4)

        # Draw coins
        for coin in self.coins:
            pygame.gfxdraw.aacircle(self.screen, coin.centerx, coin.centery, self.COIN_SIZE, self.COLOR_COIN)
            pygame.gfxdraw.filled_circle(self.screen, coin.centerx, coin.centery, self.COIN_SIZE, self.COLOR_COIN)

        # Draw particles
        for p in self.particles:
            if p['radius'] > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

        # Draw player if not game over
        if not (self.game_over and self.time_remaining > 0 and self.coins_collected_count < self.WIN_CONDITION_COINS): # Don't draw if crashed
            # Glow effect
            glow_rect = self.player_rect.inflate(10, 10)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            
            # FIX: Create a valid RGBA color tuple with desired alpha.
            # The original code `(*self.COLOR_PLAYER_GLOW, 50)` created an invalid 5-element tuple.
            glow_color = (self.COLOR_PLAYER_GLOW.r, self.COLOR_PLAYER_GLOW.g, self.COLOR_PLAYER_GLOW.b, 50)
            pygame.draw.rect(glow_surface, glow_color, glow_surface.get_rect(), border_radius=8)

            self.screen.blit(glow_surface, glow_rect.topleft)
            # Player
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=6)

    def _render_text(self, text, font, position, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect(center=position)
        shadow_rect = shadow_surf.get_rect(center=(position[0] + 2, position[1] + 2))
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        self._render_text(f"Score: {self.score}", self.font_medium, (100, 30))

        # Time
        time_sec = math.ceil(self.time_remaining / self.FPS) if self.time_remaining > 0 else 0
        time_color = self.COLOR_TEXT if time_sec > 10 else self.COLOR_OBSTACLE
        self._render_text(f"Time: {time_sec}", self.font_medium, (self.SCREEN_WIDTH - 100, 30), color=time_color)

        # Coins collected
        self._render_text(f"Coins: {self.coins_collected_count} / {self.WIN_CONDITION_COINS}", self.font_medium, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 30))

        # Game Over / Win message
        if self.game_over:
            if self.coins_collected_count >= self.WIN_CONDITION_COINS:
                msg = "YOU WIN!"
            elif self.time_remaining <= 0:
                msg = "TIME'S UP!"
            else:
                msg = "CRASHED!"
            self._render_text(msg, self.font_large, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "coins_collected": self.coins_collected_count,
            "time_remaining_steps": self.time_remaining,
        }

    def close(self):
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

        # Test observation space from get_observation
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8

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
    # This block allows you to play the game manually
    # You will need to install pygame: pip install pygame
    # And change the SDL_VIDEODRIVER to a real one
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "quartz"

    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    obs, info = env.reset()

    # Create a window to display the game
    pygame.display.set_caption("Arcade Robot Collector")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0

    print(env.user_guide)

    while not terminated:
        # --- Human Controls ---
        movement_action = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement_action, space_action, shift_action]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame. We just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()