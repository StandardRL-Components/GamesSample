import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Hold space to fire."
    )

    game_description = (
        "Survive waves of descending aliens in this retro arcade shooter. Clear all 5 waves to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.INITIAL_LIVES = 3
        self.MAX_WAVES = 5

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_PLAYER_PROJ = (255, 255, 0)
        self.COLOR_ENEMY_PROJ = (200, 0, 255)
        self.COLOR_EXPLOSION = [(255, 200, 0), (255, 128, 0), (255, 64, 0)]
        self.COLOR_STAR = (100, 100, 120)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_GAMEOVER = (255, 0, 0)
        self.COLOR_WIN = (0, 255, 0)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        # Set dummy driver for headless operation
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_main = pygame.font.Font(None, 72)

        # Game state (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.np_random = None

        self.player_pos = None
        self.player_lives = 0
        self.player_fire_cooldown = 0
        self.player_speed = 5

        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.stars = []

        self.current_wave = 1
        self.wave_transition_timer = 0
        
        # Initialize state variables
        # self.reset() is called by some gym wrappers, so it's good practice
        # to have a flag or ensure all attributes are initialized before the first reset.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - 40], dtype=np.float32)
        self.player_lives = self.INITIAL_LIVES
        self.player_fire_cooldown = 0

        self.enemies.clear()
        self.player_projectiles.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()
        
        self.current_wave = 1
        self._start_wave()

        self._spawn_stars(200)

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for each step to encourage efficiency
        
        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            reward += self._handle_collisions()
            reward += self._check_wave_completion()
        
        self.steps += 1
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS

        if self.auto_advance:
            self.clock.tick(30)

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if movement == 1:  # Up
            self.player_pos[1] -= self.player_speed
        if movement == 2:  # Down
            self.player_pos[1] += self.player_speed
        if movement == 3:  # Left
            self.player_pos[0] -= self.player_speed
        if movement == 4:  # Right
            self.player_pos[0] += self.player_speed
        
        # Clamp player position
        self.player_pos[0] = np.clip(self.player_pos[0], 10, self.WIDTH - 10)
        self.player_pos[1] = np.clip(self.player_pos[1], 10, self.HEIGHT - 10)

        if space_held and self.player_fire_cooldown <= 0:
            # sfx: player_shoot.wav
            proj_pos = self.player_pos + np.array([0, -15])
            self.player_projectiles.append(proj_pos)
            self.player_fire_cooldown = 8 # steps

    def _update_game_state(self):
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
        
        if self.wave_transition_timer > 0:
            self.wave_transition_timer -= 1

        # Update stars
        for star in self.stars:
            star[1] += star[2]  # Move star down by its speed
            if star[1] > self.HEIGHT:
                star[0] = self.np_random.integers(0, self.WIDTH)
                star[1] = 0

        # Update player projectiles
        self.player_projectiles = [p + np.array([0, -8]) for p in self.player_projectiles if p[1] > 0]

        # Update enemies and their projectiles
        enemy_fire_prob = 0.002 + (self.current_wave * 0.001)
        for enemy in self.enemies:
            enemy['pos'][1] += enemy['speed']
            if self.np_random.random() < enemy_fire_prob and self.wave_transition_timer <= 0:
                # sfx: enemy_shoot.wav
                self.enemy_projectiles.append(enemy['pos'].copy())
            if enemy['pos'][1] > self.HEIGHT + 20:
                enemy['to_remove'] = True

        # Update enemy projectiles
        self.enemy_projectiles = [p + np.array([0, 6]) for p in self.enemy_projectiles if p[1] < self.HEIGHT]

        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        self.enemies = [e for e in self.enemies if not e.get('to_remove', False)]

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 8, 20, 16)

        # Player projectiles vs enemies
        # Iterate backwards to safely remove items from a list while iterating
        for i in range(len(self.player_projectiles) - 1, -1, -1):
            proj = self.player_projectiles[i]
            proj_rect = pygame.Rect(proj[0] - 2, proj[1] - 5, 4, 10)
            
            for j in range(len(self.enemies) - 1, -1, -1):
                enemy = self.enemies[j]
                enemy_rect = pygame.Rect(enemy['pos'][0] - 12, enemy['pos'][1] - 12, 24, 24)
                
                if proj_rect.colliderect(enemy_rect):
                    # sfx: explosion.wav
                    self._create_explosion(enemy['pos'])
                    self.enemies.pop(j)
                    self.player_projectiles.pop(i)
                    self.score += 10
                    reward += 0.1
                    break  # Projectile is used up, move to next projectile

        # Enemy projectiles vs player
        # Iterate backwards to safely remove items
        for i in range(len(self.enemy_projectiles) - 1, -1, -1):
            proj = self.enemy_projectiles[i]
            proj_rect = pygame.Rect(proj[0] - 3, proj[1] - 3, 6, 6)
            
            if player_rect.colliderect(proj_rect):
                self.enemy_projectiles.pop(i)
                self.player_lives -= 1
                # sfx: player_hit.wav
                self._create_explosion(self.player_pos, count=30)
                if self.player_lives <= 0:
                    self.game_over = True
                    reward -= 100
                break

        # Enemies vs player
        # Iterate backwards to safely remove items
        for i in range(len(self.enemies) - 1, -1, -1):
            enemy = self.enemies[i]
            enemy_rect = pygame.Rect(enemy['pos'][0] - 12, enemy['pos'][1] - 12, 24, 24)
            
            if player_rect.colliderect(enemy_rect):
                self.enemies.pop(i)
                self.player_lives -= 1
                self._create_explosion(enemy['pos'])
                self._create_explosion(self.player_pos, count=30)
                if self.player_lives <= 0:
                    self.game_over = True
                    reward -= 100
                break
        return reward
    
    def _check_wave_completion(self):
        if not self.enemies and not self.game_over and self.wave_transition_timer <= 0:
            if self.current_wave >= self.MAX_WAVES:
                self.win = True
                self.game_over = True
                return 100 # Win reward
            else:
                self.current_wave += 1
                self._start_wave()
                return 1 # Wave clear reward
        return 0

    def _start_wave(self):
        self.wave_transition_timer = 60 # 2 seconds at 30fps
        num_enemies = 8 + self.current_wave * 2
        rows = (num_enemies // 8) + 1
        for i in range(num_enemies):
            row = i // 8
            col = i % 8
            x = self.WIDTH * (col + 1) / 9
            y = -30 - row * 40
            self.enemies.append({
                'pos': np.array([x, y], dtype=np.float32),
                'speed': 0.5 + self.current_wave * 0.1,
            })

    def _spawn_stars(self, count):
        self.stars.clear()
        for _ in range(count):
            self.stars.append([
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.uniform(0.5, 2.0) # speed
            ])
            
    def _create_explosion(self, pos, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'color': random.choice(self.COLOR_EXPLOSION)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, speed in self.stars:
            size = int(speed / 2)
            pygame.draw.rect(self.screen, self.COLOR_STAR, (int(x), int(y), size, size))
            
        # Enemy projectiles
        for p in self.enemy_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), 4, self.COLOR_ENEMY_PROJ)
            pygame.gfxdraw.aacircle(self.screen, int(p[0]), int(p[1]), 4, self.COLOR_ENEMY_PROJ)

        # Player projectiles
        for p in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, (int(p[0] - 2), int(p[1] - 8), 4, 16))

        # Enemies
        for enemy in self.enemies:
            x, y = int(enemy['pos'][0]), int(enemy['pos'][1])
            pts = [(x, y-12), (x+12, y+12), (x-12, y+12)]
            pygame.gfxdraw.aapolygon(self.screen, pts, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, pts, self.COLOR_ENEMY)

        # Player
        if self.player_lives > 0:
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            player_pts = [(px, py-12), (px-10, py+8), (px+10, py+8)]
            pygame.gfxdraw.aapolygon(self.screen, player_pts, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, player_pts, self.COLOR_PLAYER)
            # Engine flame
            if self.np_random.random() > 0.3:
                flame_h = self.np_random.integers(5, 12)
                flame_pts = [(px, py+8), (px-4, py+8+flame_h), (px+4, py+8+flame_h)]
                pygame.gfxdraw.aapolygon(self.screen, flame_pts, self.COLOR_PLAYER_PROJ)

        # Particles
        for p in self.particles:
            size = max(0, int(p['life'] / 4))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)
    
    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 10))

        # Wave
        wave_surf = self.font_ui.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_UI)
        self.screen.blit(wave_surf, (self.WIDTH // 2 - wave_surf.get_width() // 2, 10))

        # Lives
        for i in range(self.player_lives):
            px, py = self.WIDTH - 20 - (i * 25), 20
            player_pts = [(px, py-8), (px-6, py+5), (px+6, py+5)]
            pygame.gfxdraw.filled_polygon(self.screen, player_pts, self.COLOR_PLAYER)

        # Wave transition text
        if self.wave_transition_timer > 0 and not self.game_over:
            wave_text = self.font_main.render(f"WAVE {self.current_wave}", True, self.COLOR_UI)
            text_rect = wave_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(wave_text, text_rect)

        # Game Over / Win text
        if self.game_over:
            if self.win:
                msg, color = "YOU WIN!", self.COLOR_WIN
            else:
                msg, color = "GAME OVER", self.COLOR_GAMEOVER
            
            end_surf = self.font_main.render(msg, True, color)
            text_rect = end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.current_wave,
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # Validation check (optional, but good for development)
    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env)
        print("✓ Gymnasium environment check passed")
    except Exception as e:
        print(f"✗ Gymnasium environment check failed: {e}")


    obs, info = env.reset()
    print("Initial state:", info)

    terminated = False
    truncated = False
    total_reward = 0
    for i in range(2000):
        action = env.action_space.sample() # Take random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i+1) % 100 == 0:
            print(f"Step {i+1}: Info={info}, Reward={reward:.2f}, Total Reward={total_reward:.2f}")
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps. Final Info: {info}, Total Reward: {total_reward:.2f}")
            break
    
    env.close()