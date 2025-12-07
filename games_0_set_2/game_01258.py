
# Generated: 2025-08-27T16:32:35.081389
# Source Brief: brief_01258.md
# Brief Index: 1258

        
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
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your ship and dodge the incoming projectiles."
    )

    game_description = (
        "Survive a relentless barrage of alien projectiles for 60 seconds in this top-down arcade "
        "shooter. Dodge to survive!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = self.FPS * 60  # 60 seconds

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (50, 255, 150, 60)
        self.COLOR_PROJECTILE = (255, 80, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_GAMEOVER = (255, 50, 50)
        self.COLOR_WIN = (50, 255, 50)

        # Player settings
        self.PLAYER_SIZE = 12
        self.PLAYER_SPEED = 5.0
        self.INITIAL_LIVES = 3

        # Projectile settings
        self.INITIAL_SPAWN_RATE = 1.0  # projectiles per second
        self.SPAWN_RATE_INCREASE = 0.05
        self.SPAWN_RATE_INTERVAL = self.FPS * 10 # every 10 seconds
        self.PROJECTILE_SPEED_MIN = 2.0
        self.PROJECTILE_SPEED_MAX = 4.0
        self.PROJECTILE_SIZE = 6
        self.MAX_PROJECTILES = 50

        # Reward settings
        self.REWARD_SURVIVE = 0.1
        self.REWARD_HIT = -10.0
        self.REWARD_WIN = 100.0
        self.REWARD_STATIONARY_PENALTY = -0.2
        self.STATIONARY_RADIUS = 50
        self.STATIONARY_STEPS_THRESHOLD = 10

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
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 36)

        # State variables (initialized in reset)
        self.player_pos = None
        self.player_lives = None
        self.projectiles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.projectile_spawn_rate = None
        self.last_pos_history = None
        
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.display = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Arcade Survivor")
            
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_lives = self.INITIAL_LIVES
        self.projectiles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.projectile_spawn_rate = self.INITIAL_SPAWN_RATE
        self.last_pos_history = deque(maxlen=self.STATIONARY_STEPS_THRESHOLD)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = 0.0

        if not self.game_over:
            # --- Update Game Logic ---
            self._handle_player_movement(movement)
            hit_count = self._update_projectiles()
            self._spawn_projectiles()
            self._update_difficulty()

            # --- Calculate Reward ---
            reward += self.REWARD_SURVIVE
            self.score += self.REWARD_SURVIVE

            if hit_count > 0:
                self.player_lives -= hit_count
                reward += self.REWARD_HIT * hit_count
                self.score += self.REWARD_HIT * hit_count
                # Sound: Player Hit

            # Stationary penalty
            self.last_pos_history.append(self.player_pos.copy())
            if (len(self.last_pos_history) == self.STATIONARY_STEPS_THRESHOLD and
                self.player_pos.distance_to(self.last_pos_history[0]) < self.STATIONARY_RADIUS):
                reward += self.REWARD_STATIONARY_PENALTY
                # Note: score does not get penalized, only the agent's reward
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and self.win:
            reward += self.REWARD_WIN
            self.score += self.REWARD_WIN

        if self.render_mode == "human":
            self.render()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1:  # Up
            move_vec.y = -1
        elif movement == 2:  # Down
            move_vec.y = 1
        elif movement == 3:  # Left
            move_vec.x = -1
        elif movement == 4:  # Right
            move_vec.x = 1

        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
        
        # Boundary checks
        self.player_pos.x = max(self.PLAYER_SIZE, min(self.player_pos.x, self.WIDTH - self.PLAYER_SIZE))
        self.player_pos.y = max(self.PLAYER_SIZE, min(self.player_pos.y, self.HEIGHT - self.PLAYER_SIZE))

    def _spawn_projectiles(self):
        if len(self.projectiles) >= self.MAX_PROJECTILES:
            return

        spawn_prob = self.projectile_spawn_rate / self.FPS
        if self.np_random.random() < spawn_prob:
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -self.PROJECTILE_SIZE)
            elif edge == 1: # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.PROJECTILE_SIZE)
            elif edge == 2: # Left
                pos = pygame.Vector2(-self.PROJECTILE_SIZE, self.np_random.uniform(0, self.HEIGHT))
            else: # Right
                pos = pygame.Vector2(self.WIDTH + self.PROJECTILE_SIZE, self.np_random.uniform(0, self.HEIGHT))
            
            # Aim towards the center with some randomness
            target = pygame.Vector2(self.WIDTH/2, self.HEIGHT/2)
            target.x += self.np_random.uniform(-self.WIDTH/4, self.WIDTH/4)
            target.y += self.np_random.uniform(-self.HEIGHT/4, self.HEIGHT/4)
            
            vel = (target - pos).normalize() * self.np_random.uniform(self.PROJECTILE_SPEED_MIN, self.PROJECTILE_SPEED_MAX)

            self.projectiles.append({
                "pos": pos, 
                "vel": vel,
                "trail": deque(maxlen=10)
            })
            # Sound: Projectile Spawn

    def _update_projectiles(self):
        hit_count = 0
        projectiles_to_keep = []
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        for p in self.projectiles:
            p['trail'].append(p['pos'].copy())
            p['pos'] += p['vel']
            
            proj_rect = pygame.Rect(p['pos'].x - self.PROJECTILE_SIZE / 2, p['pos'].y - self.PROJECTILE_SIZE / 2, self.PROJECTILE_SIZE, self.PROJECTILE_SIZE)
            
            is_hit = player_rect.colliderect(proj_rect)
            is_offscreen = not self.screen.get_rect().inflate(50, 50).colliderect(proj_rect)

            if is_hit:
                hit_count += 1
            elif not is_offscreen:
                projectiles_to_keep.append(p)

        self.projectiles = projectiles_to_keep
        return hit_count

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.SPAWN_RATE_INTERVAL == 0:
            self.projectile_spawn_rate += self.SPAWN_RATE_INCREASE

    def _check_termination(self):
        if self.game_over:
            return True
            
        if self.player_lives <= 0:
            self.game_over = True
            self.win = False
            # Sound: Game Over
            return True
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = True
            # Sound: Victory
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render projectiles and trails
        for p in self.projectiles:
            # Trail
            for i, pos in enumerate(p['trail']):
                alpha = int(255 * (i / len(p['trail'])))
                color = (*self.COLOR_PROJECTILE, alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(self.PROJECTILE_SIZE / 2), color)
            # Head
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), self.PROJECTILE_SIZE, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), self.PROJECTILE_SIZE, self.COLOR_PROJECTILE)

        # Render player
        if self.player_lives > 0:
            # Glow effect
            glow_radius = int(self.PLAYER_SIZE * 2.5)
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), glow_radius, self.COLOR_PLAYER_GLOW)
            # Player square
            player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
            player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)
    
    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_medium.render(f"Lives: {max(0, self.player_lives)}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = self.font_medium.render(f"Time: {max(0, time_left):.1f}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(centerx=self.WIDTH / 2, y=10)
        self.screen.blit(timer_text, timer_rect)

        # Game over / Win message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_GAMEOVER
            
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def render(self):
        if self.render_mode == "human":
            if self.display is None:
                pygame.display.init()
                self.display = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            
            # The observation is already the rendered screen
            obs_surface = pygame.surfarray.make_surface(np.transpose(self._get_observation(), (1, 0, 2)))
            self.display.blit(obs_surface, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "win": self.win,
        }

    def close(self):
        if self.render_mode == "human" and self.display is not None:
            pygame.display.quit()
            self.display = None
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage for human play
if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not done:
        # Default action is no-op
        movement = 0
        
        # Check for key presses
        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize first key found (e.g., up over down)

        action = [movement, 0, 0] # Space and Shift are not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Allow quitting the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Game Over! Final Info: {info}")
    
    # Keep the window open for a few seconds to show the final screen
    if info.get('win', False) or info.get('lives', 1) <= 0:
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < 3000:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
    
    env.close()