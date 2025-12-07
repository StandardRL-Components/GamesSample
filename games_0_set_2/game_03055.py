import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold shift for a temporary shield. Press space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive for two minutes against waves of descending alien fighters in this retro top-down shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 120
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        
        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 128, 64)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (128, 25, 25)
        self.COLOR_PLAYER_PROJ = (255, 255, 0)
        self.COLOR_ENEMY_PROJ = (255, 100, 0)
        self.COLOR_SHIELD = (100, 150, 255, 100) # RGBA
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_EXPLOSION = (255, 200, 150)
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Player state
        self.player_pos = np.array([0.0, 0.0])
        self.player_speed = 6.0
        self.player_size = 12
        self.player_fire_cooldown = 0
        
        # Shield state
        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown = 0
        self.shield_duration_frames = int(0.5 * self.FPS)
        self.shield_cooldown_frames = int(3.0 * self.FPS)

        # Game state
        self.player_projectiles = []
        self.enemies = []
        self.enemy_projectiles = []
        self.particles = []
        self.stars = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.enemy_spawn_timer = 0
        self.base_enemy_speed = 1.0
        
        # Initialize state variables
        # self.reset() is called by the wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = np.array([self.WIDTH / 2.0, self.HEIGHT - 50.0])
        self.player_fire_cooldown = 0
        
        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown = 0
        
        self.player_projectiles = []
        self.enemies = []
        self.enemy_projectiles = []
        self.particles = []
        
        self.enemy_spawn_timer = self.FPS * 2 # First wave in 2 seconds

        # Create a static starfield
        if not self.stars:
            for _ in range(200):
                self.stars.append({
                    "pos": np.array([self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)]),
                    "speed": self.np_random.uniform(0.2, 0.8),
                    "size": self.np_random.integers(1, 3)
                })

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            return (
                self._get_observation(),
                0, # No reward after game over
                True,
                False,
                self._get_info()
            )

        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1
        
        # --- Update Timers ---
        self.player_fire_cooldown = max(0, self.player_fire_cooldown - 1)
        self.shield_timer = max(0, self.shield_timer - 1)
        self.shield_cooldown = max(0, self.shield_cooldown - 1)
        self.enemy_spawn_timer = max(0, self.enemy_spawn_timer - 1)
        
        if self.shield_timer == 0:
            self.shield_active = False
            
        # --- Handle Player Input ---
        # Movement
        if movement == 1: self.player_pos[1] -= self.player_speed
        if movement == 2: self.player_pos[1] += self.player_speed
        if movement == 3: self.player_pos[0] -= self.player_speed
        if movement == 4: self.player_pos[0] += self.player_speed
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_size, self.WIDTH - self.player_size)
        self.player_pos[1] = np.clip(self.player_pos[1], self.player_size, self.HEIGHT - self.player_size)

        # Shield
        if shift_pressed and self.shield_cooldown == 0:
            self.shield_active = True
            self.shield_timer = self.shield_duration_frames
            self.shield_cooldown = self.shield_cooldown_frames
            self._create_effect(self.player_pos, self.COLOR_SHIELD, 20, 10, 'burst')

        # Fire
        if space_pressed and self.player_fire_cooldown == 0:
            self.player_projectiles.append(np.copy(self.player_pos) - np.array([0, self.player_size]))
            self.player_fire_cooldown = self.FPS // 5 # 5 shots per second

        # --- Game Logic ---
        self._update_stars()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        # --- Enemy Spawning ---
        if self.enemy_spawn_timer == 0:
            self._spawn_wave()
            spawn_delay = self.np_random.uniform(2, 5) * (1 - 0.7 * (self.steps / self.MAX_STEPS))
            self.enemy_spawn_timer = int(spawn_delay * self.FPS)
            
        # --- Collision Detection ---
        # Player projectiles vs Enemies
        hit_projectiles = set()
        hit_enemies = set()

        for i, proj in enumerate(self.player_projectiles):
            for j, enemy in enumerate(self.enemies):
                if j in hit_enemies:
                    continue
                if np.linalg.norm(proj - enemy['pos']) < enemy['size'] + 4:
                    hit_projectiles.add(i)
                    hit_enemies.add(j)
                    self.score += 10
                    reward += 1.0
                    self._create_effect(enemy['pos'], self.COLOR_EXPLOSION, 30, 15, 'explosion')
                    break
        
        if hit_enemies:
            self.enemies = [e for i, e in enumerate(self.enemies) if i not in hit_enemies]
        if hit_projectiles:
            self.player_projectiles = [p for i, p in enumerate(self.player_projectiles) if i not in hit_projectiles]

        # Enemy projectiles vs Player
        proj_to_remove_idx = -1
        for i, proj in enumerate(self.enemy_projectiles):
            if np.linalg.norm(proj - self.player_pos) < self.player_size + 4:
                proj_to_remove_idx = i
                if self.shield_active:
                    reward -= 0.5
                    self._create_effect(self.player_pos, self.COLOR_SHIELD, 15, 10, 'burst')
                else:
                    self.game_over = True
                    reward -= 100.0
                    self._create_effect(self.player_pos, self.COLOR_PLAYER, 50, 20, 'explosion')
                break
        
        if proj_to_remove_idx != -1:
            self.enemy_projectiles.pop(proj_to_remove_idx)

        # --- Termination Checks ---
        terminated = self.game_over
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100.0

        if not terminated:
            reward += 0.01 # Small survival reward per step

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _spawn_wave(self):
        num_enemies = self.np_random.integers(3, 7)
        start_x = self.np_random.uniform(50, self.WIDTH - 50 * num_enemies)
        start_y = -20.0
        spacing = self.np_random.uniform(40, 60)
        
        for i in range(num_enemies):
            self.enemies.append({
                'pos': np.array([start_x + i * spacing, start_y]),
                'size': self.np_random.integers(10, 15),
                'fire_timer': self.np_random.integers(0, self.FPS)
            })

    def _update_enemies(self):
        progress = self.steps / self.MAX_STEPS
        current_enemy_speed = self.base_enemy_speed + 1.5 * progress
        fire_rate_chance = min(2.0, 0.5 + 1.5 * progress) / self.FPS
        
        kept_enemies = []
        for enemy in self.enemies:
            enemy['pos'][1] += current_enemy_speed
            if enemy['pos'][1] > self.HEIGHT + enemy['size']:
                continue

            enemy['fire_timer'] -= 1
            if enemy['fire_timer'] <= 0 and self.np_random.random() < fire_rate_chance:
                self.enemy_projectiles.append(np.copy(enemy['pos']))
                enemy['fire_timer'] = self.np_random.integers(self.FPS // 2, self.FPS * 2)
            
            kept_enemies.append(enemy)
        self.enemies = kept_enemies

    def _update_projectiles(self):
        player_proj_speed = 10.0
        enemy_proj_speed = 5.0
        
        self.player_projectiles = [p - np.array([0, player_proj_speed]) for p in self.player_projectiles if p[1] > -10]
        self.enemy_projectiles = [p + np.array([0, enemy_proj_speed]) for p in self.enemy_projectiles if p[1] < self.HEIGHT + 10]

    def _update_stars(self):
        for star in self.stars:
            star['pos'][1] += star['speed']
            if star['pos'][1] > self.HEIGHT:
                star['pos'][0] = self.np_random.uniform(0, self.WIDTH)
                star['pos'][1] = 0

    def _update_particles(self):
        kept_particles = []
        for p in self.particles:
            p['lifetime'] -= 1
            if p['lifetime'] > 0:
                p['pos'] += p['vel']
                p['radius'] -= p['shrink']
                kept_particles.append(p)
        self.particles = kept_particles

    def _create_effect(self, pos, color, count, lifetime, effect_type):
        for _ in range(count):
            if effect_type == 'explosion':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 5)
                vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
                radius = self.np_random.uniform(5, 15)
                shrink = radius / lifetime
            elif effect_type == 'burst':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(0.5, 2)
                vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
                radius = self.np_random.uniform(2, 5)
                shrink = radius / lifetime
            
            self.particles.append({
                'pos': np.copy(pos),
                'vel': vel,
                'radius': radius,
                'lifetime': self.np_random.integers(lifetime // 2, lifetime),
                'color': color,
                'shrink': shrink
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Stars
        for star in self.stars:
            pygame.draw.circle(self.screen, (255,255,255), star['pos'].astype(int), star['size'])

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 20))
            color = p['color'][:3]
            radius = max(0, int(p['radius']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, (*color, alpha))
        
        # Enemy projectiles
        for proj in self.enemy_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_PROJ, (int(proj[0]-2), int(proj[1]-4), 4, 8))
            
        # Player projectiles
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, (int(proj[0]-2), int(proj[1]-6), 4, 12))

        # Enemies
        for enemy in self.enemies:
            pos, size = enemy['pos'], enemy['size']
            points = [
                (pos[0], pos[1] + size),
                (pos[0] - size/2, pos[1] - size/2),
                (pos[0] + size/2, pos[1] - size/2)
            ]
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_ENEMY)
            pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_ENEMY_GLOW)

        # Player
        if not (self.game_over and not self.win):
            pos, size = self.player_pos, self.player_size
            points = [
                (pos[0], pos[1] - size),
                (pos[0] - size * 0.8, pos[1] + size * 0.8),
                (pos[0] + size * 0.8, pos[1] + size * 0.8)
            ]
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_PLAYER)
            pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_PLAYER_GLOW)
            
            # Shield
            if self.shield_active:
                alpha = int(100 + 100 * (self.shield_timer / self.shield_duration_frames))
                color = (*self.COLOR_SHIELD[:3], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(size*2.5), color)
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(size*2.5), color)


    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Time
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_surf = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))
        
        # Shield Cooldown Indicator
        if self.shield_cooldown > 0:
            bar_width = 100
            bar_height = 10
            fill_ratio = self.shield_cooldown / self.shield_cooldown_frames
            pygame.draw.rect(self.screen, (50,50,80), (10, self.HEIGHT - 20, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_SHIELD, (10, self.HEIGHT - 20, bar_width * (1-fill_ratio), bar_height))

        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "YOU WIN"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_ENEMY
            
            over_surf = self.font_game_over.render(msg, True, color)
            over_rect = over_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(over_surf, over_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": max(0, (self.MAX_STEPS - self.steps) / self.FPS),
            "win": self.win
        }
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # You will need to install pygame: pip install pygame
    env = GameEnv()
    obs, info = env.reset()
    
    # Switch to a visible driver
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.quit() # Quit the dummy driver
    pygame.init() # Re-init with visible driver
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Space Survivor")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # We need to re-render the initial state to the new visible screen
    obs, info = env.reset()
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

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
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)

        clock.tick(env.FPS)
        
    env.close()