
# Generated: 2025-08-27T12:49:39.702248
# Source Brief: brief_00174.md
# Brief Index: 174

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A top-down arcade game where the player must survive a zombie horde for 60 seconds.
    The player collects coins for points while dodging zombies that constantly move towards them.
    The game ends if the player is caught by a zombie or survives the full duration.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move and avoid the zombies."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a zombie horde for 60 seconds by dodging and collecting coins in a top-down arcade environment."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (50, 200, 50)
        self.COLOR_ZOMBIE = (200, 50, 50)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_UI = (230, 230, 230)
        self.COLOR_PARTICLE = (255, 240, 100)

        # Game parameters
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 4  # Increased for better game feel
        self.ZOMBIE_SIZE = 16
        self.ZOMBIE_SPEED = 1
        self.NUM_ZOMBIES = 20
        self.COIN_RADIUS = 8
        self.INITIAL_COINS = 10
        self.COIN_SPAWN_INTERVAL = 5 * self.FPS  # 5 seconds
        self.MAX_COINS = 15

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_end = pygame.font.Font(None, 72)

        # --- State Variables (initialized in reset) ---
        self.player_rect = None
        self.zombies = None
        self.coins = None
        self.particles = None
        self.score = None
        self.steps = None
        self.terminated = None

        if render_mode == "human":
            pygame.display.set_caption("Zombie Survival")
            self.human_screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_rect = pygame.Rect(
            self.SCREEN_WIDTH // 2 - self.PLAYER_SIZE // 2,
            self.SCREEN_HEIGHT // 2 - self.PLAYER_SIZE // 2,
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        self.zombies = [self._spawn_zombie() for _ in range(self.NUM_ZOMBIES)]
        self.coins = [self._spawn_coin() for _ in range(self.INITIAL_COINS)]
        self.particles = []
        self.score = 0
        self.steps = 0
        self.terminated = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, self.terminated, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Action Handling ---
        movement = action[0]
        player_moved = self._move_player(movement)

        # --- Game Logic ---
        self._update_zombies()
        self._update_particles()
        
        # --- Rewards & Events ---
        # Continuous survival reward
        reward += 0.1
        if not player_moved:
            reward -= 0.01  # Penalty for inactivity

        # Coin collection
        collected_coin_index = self.player_rect.collidelist(self.coins)
        if collected_coin_index != -1:
            coin_rect = self.coins.pop(collected_coin_index)
            self.score += 1
            reward += 1.0
            self._create_particles(coin_rect.center)
            # sfx_coin_collect

        # Coin spawning
        if self.steps > 0 and self.steps % self.COIN_SPAWN_INTERVAL == 0:
            if len(self.coins) < self.MAX_COINS:
                self.coins.append(self._spawn_coin())

        # --- Termination Check ---
        # Zombie collision (failure)
        if self.player_rect.collidelist(self.zombies) != -1:
            self.terminated = True
            reward = -10.0  # Override other rewards on death
            # sfx_player_death

        # Time limit reached (victory)
        if self.steps >= self.MAX_STEPS:
            self.terminated = True
            reward = 100.0  # Override other rewards on win
            # sfx_win_game

        return self._get_observation(), reward, self.terminated, False, self._get_info()

    def _move_player(self, movement):
        moved = True
        if movement == 1:  # Up
            self.player_rect.y -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_rect.y += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_rect.x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_rect.x += self.PLAYER_SPEED
        else:  # No-op
            moved = False

        # Clamp player to screen bounds
        self.player_rect.left = max(0, self.player_rect.left)
        self.player_rect.right = min(self.SCREEN_WIDTH, self.player_rect.right)
        self.player_rect.top = max(0, self.player_rect.top)
        self.player_rect.bottom = min(self.SCREEN_HEIGHT, self.player_rect.bottom)
        return moved

    def _update_zombies(self):
        player_center = self.player_rect.center
        for zombie in self.zombies:
            dx = player_center[0] - zombie.centerx
            dy = player_center[1] - zombie.centery
            dist = math.hypot(dx, dy)
            if dist > 0:
                zombie.x += (dx / dist) * self.ZOMBIE_SPEED
                zombie.y += (dy / dist) * self.ZOMBIE_SPEED

            # Clamp zombies to screen bounds (prevents them from leaving)
            zombie.left = max(0, zombie.left)
            zombie.right = min(self.SCREEN_WIDTH, zombie.right)
            zombie.top = max(0, zombie.top)
            zombie.bottom = min(self.SCREEN_HEIGHT, zombie.bottom)

    def _spawn_zombie(self):
        while True:
            x = self.np_random.integers(0, self.SCREEN_WIDTH - self.ZOMBIE_SIZE)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT - self.ZOMBIE_SIZE)
            rect = pygame.Rect(x, y, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            # Ensure zombies don't spawn too close to the player's start position
            if not rect.colliderect(self.player_rect.inflate(100, 100)):
                return rect

    def _spawn_coin(self):
        x = self.np_random.integers(self.COIN_RADIUS, self.SCREEN_WIDTH - self.COIN_RADIUS)
        y = self.np_random.integers(self.COIN_RADIUS, self.SCREEN_HEIGHT - self.COIN_RADIUS)
        return pygame.Rect(x - self.COIN_RADIUS, y - self.COIN_RADIUS, self.COIN_RADIUS * 2, self.COIN_RADIUS * 2)

    def _create_particles(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': [vx, vy], 'life': lifetime, 'max_life': lifetime})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # This is the main rendering loop
        self.screen.fill(self.COLOR_BG)
        
        # Render particles
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            size = max(1, int(5 * life_ratio))
            pygame.draw.rect(self.screen, self.COLOR_PARTICLE, (int(p['pos'][0]), int(p['pos'][1]), size, size))
            
        # Render coins
        for coin in self.coins:
            pygame.draw.circle(self.screen, self.COLOR_COIN, coin.center, self.COIN_RADIUS)
        
        # Render zombies
        for zombie in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, zombie)

        # Render player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)
        
        # Render UI
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_surf = self.font_ui.render(f"Time: {time_left:.1f}", True, self.COLOR_UI)
        timer_rect = timer_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_surf, timer_rect)

        # Game Over / Win message
        if self.terminated:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            won = self.steps >= self.MAX_STEPS
            msg = "YOU SURVIVED!" if won else "GAME OVER"
            color = (100, 255, 150) if won else (255, 80, 80)
            
            end_surf = self.font_end.render(msg, True, color)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_surf, end_rect)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
        elif self.render_mode == "human":
            # In human mode, we get the observation and blit it to the display
            obs_array = self._get_observation()
            # The observation is (H, W, C) but pygame wants (W, H, C) for surfarray.make_surface
            # And our internal self.screen is already what we need.
            # We just need to transpose it back for display if we use the obs_array.
            # It's easier to just blit self.screen directly.
            surf_to_show = pygame.transform.rotate(pygame.transform.flip(self.screen, True, False), 90)
            self.human_screen.blit(surf_to_show, (0,0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.FPS)
            return obs_array

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="human")
    env.validate_implementation()
    
    obs, info = env.reset()
    terminated = False
    
    # Mapping from Pygame keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not terminated:
        # Default action is no-op
        movement = 0
        
        # Poll for keyboard events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize first key found (e.g., up over down)

        # Space and Shift are not used in this game
        space_held = 0
        shift_held = 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()