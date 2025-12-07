
# Generated: 2025-08-27T19:54:24.620355
# Source Brief: brief_02288.md
# Brief Index: 2288

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move your ship. Hold space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your planetary base from waves of descending alien invaders in this side-scrolling space shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 180

        # Colors
        self.COLOR_BG = (16, 16, 32)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_ENEMY_1 = (255, 50, 50)
        self.COLOR_ENEMY_2 = (255, 150, 50)
        self.COLOR_ENEMY_3 = (255, 75, 200)
        self.COLOR_ENEMY_GLOW = (255, 50, 50, 50)
        self.COLOR_PLAYER_PROJ = (100, 200, 255)
        self.COLOR_ENEMY_PROJ = (255, 100, 100)
        self.COLOR_POWERUP_SHIELD = (50, 150, 255)
        self.COLOR_POWERUP_FIRERATE = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_BAR = (0, 200, 100)
        self.COLOR_UI_BAR_BG = (100, 0, 0)
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.rng = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_timer = 0.0
        
        self.player = None
        self.player_projectiles = []
        self.player_fire_cooldown = 0
        self.player_powerup = {'type': 'none', 'duration': 0}

        self.base_health = 100
        self.base_rect = None

        self.enemies = []
        self.enemy_projectiles = []
        self.enemy_spawn_timer = 0
        self.base_spawn_rate = 0.5 # enemies per second
        self.base_enemy_proj_speed = 150

        self.powerups = []
        self.particles = []
        self.stars = []
        
        self.screen_shake = 0
        
        # Initialize state variables
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_timer = 0.0
        
        self.player = pygame.Rect(self.WIDTH // 2 - 15, self.HEIGHT - 80, 30, 20)
        self.player_projectiles.clear()
        self.player_fire_cooldown = 0
        self.player_powerup = {'type': 'none', 'duration': 0}

        self.base_health = 100
        self.base_rect = pygame.Rect(0, self.HEIGHT - 30, self.WIDTH, 30)

        self.enemies.clear()
        self.enemy_projectiles.clear()
        self.enemy_spawn_timer = 2.0 # Initial delay

        self.powerups.clear()
        self.particles.clear()
        
        self.screen_shake = 0

        # Create starfield
        self.stars.clear()
        for _ in range(150):
            self.stars.append({
                'pos': pygame.Vector2(self.rng.integers(0, self.WIDTH), self.rng.integers(0, self.HEIGHT)),
                'speed': self.rng.uniform(5, 20),
                'size': self.rng.integers(1, 3),
                'color': self.rng.integers(50, 150)
            })
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        dt = self.clock.tick(self.FPS) / 1000.0
        self.steps += 1
        self.game_timer += dt
        reward = 0.001 # Small survival reward

        # --- Handle Input & Update Player ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        player_moved = False
        if movement == 3: # Left
            self.player.x -= 300 * dt
            player_moved = True
        elif movement == 4: # Right
            self.player.x += 300 * dt
            player_moved = True

        self.player.left = max(0, self.player.left)
        self.player.right = min(self.WIDTH, self.player.right)

        if not player_moved:
            reward -= 0.01 # Penalty for standing still

        # Cooldowns
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= dt
        if self.player_powerup['duration'] > 0:
            self.player_powerup['duration'] -= dt
        else:
            self.player_powerup['type'] = 'none'

        # Shooting
        fire_rate = 0.2 if self.player_powerup['type'] != 'firerate' else 0.08
        if space_held and self.player_fire_cooldown <= 0:
            # Sfx: Player shoot
            proj_pos = (self.player.centerx - 2, self.player.top)
            self.player_projectiles.append(pygame.Rect(proj_pos[0], proj_pos[1], 4, 12))
            self.player_fire_cooldown = fire_rate
            self._create_particles(pygame.Vector2(self.player.centerx, self.player.top), self.COLOR_PLAYER, 5, 1, 50)

        # --- Update Game World ---
        self._update_difficulty()
        self._update_spawns(dt)
        self._update_entities(dt)
        
        # --- Handle Collisions ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- Check Termination ---
        terminated = False
        if self.base_health <= 0:
            reward = -100
            terminated = True
            self.game_over = True
        elif self.game_timer >= self.GAME_DURATION_SECONDS:
            reward = 100
            terminated = True
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _update_difficulty(self):
        self.current_spawn_rate = self.base_spawn_rate + 0.05 * self.game_timer
        self.current_spawn_rate = min(self.current_spawn_rate, 2.0) # Cap at 2/sec
        
        difficulty_tier_bonus = self.game_timer // 30
        self.current_enemy_proj_speed = self.base_enemy_proj_speed + (difficulty_tier_bonus * 0.1 * self.base_enemy_proj_speed)

    def _update_spawns(self, dt):
        # Enemy spawning
        self.enemy_spawn_timer -= dt
        if self.enemy_spawn_timer <= 0:
            spawn_x = self.rng.integers(20, self.WIDTH - 20)
            
            # Introduce new types over time
            tier = 0
            if self.game_timer > 120: tier = self.rng.choice([1,2,3], p=[0.3, 0.4, 0.3])
            elif self.game_timer > 60: tier = self.rng.choice([1,2], p=[0.5, 0.5])
            else: tier = 1
            
            self.enemies.append({
                'rect': pygame.Rect(spawn_x, -30, 25, 25),
                'type': tier,
                'fire_cooldown': self.rng.uniform(1.0, 3.0),
                'phase': self.rng.uniform(0, 2 * math.pi) # For sine wave movement
            })
            self.enemy_spawn_timer = 1.0 / self.current_spawn_rate

    def _update_entities(self, dt):
        # Move projectiles
        for proj in self.player_projectiles[:]:
            proj.y -= 500 * dt
            if proj.bottom < 0:
                self.player_projectiles.remove(proj)
        
        for proj in self.enemy_projectiles[:]:
            proj.y += self.current_enemy_proj_speed * dt
            if proj.top > self.HEIGHT:
                self.enemy_projectiles.remove(proj)

        # Move enemies and handle their shooting
        for enemy in self.enemies[:]:
            enemy['fire_cooldown'] -= dt
            
            # Movement patterns
            if enemy['type'] == 1: # Straight down
                enemy['rect'].y += 80 * dt
            elif enemy['type'] == 2: # Zig-zag
                enemy['rect'].y += 60 * dt
                enemy['rect'].x += 100 * math.sin(self.game_timer * 3 + enemy['phase']) * dt
            elif enemy['type'] == 3: # Sine wave
                enemy['rect'].y += 70 * dt
                enemy['rect'].x = self.WIDTH/2 + (self.WIDTH/2 - 30) * math.sin(self.game_timer * 0.7 + enemy['phase'])

            # Shooting
            if enemy['fire_cooldown'] <= 0:
                # Sfx: Enemy shoot
                proj_pos = (enemy['rect'].centerx - 2, enemy['rect'].bottom)
                self.enemy_projectiles.append(pygame.Rect(proj_pos[0], proj_pos[1], 4, 10))
                enemy['fire_cooldown'] = self.rng.uniform(2.0, 4.0) / (self.current_spawn_rate / self.base_spawn_rate)

            if enemy['rect'].top > self.HEIGHT:
                self.enemies.remove(enemy)
                self.base_health -= 15 # Penalty for letting one through
                self.screen_shake = 20
        
        # Move powerups
        for powerup in self.powerups[:]:
            powerup['rect'].y += 100 * dt
            if powerup['rect'].top > self.HEIGHT:
                self.powerups.remove(powerup)

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel'] * dt
            p['vel'] *= 0.95
            p['lifetime'] -= dt
            if p['lifetime'] <= 0:
                self.particles.remove(p)
                
        # Update screen shake
        if self.screen_shake > 0:
            self.screen_shake -= 40 * dt
            self.screen_shake = max(0, self.screen_shake)

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if proj.colliderect(enemy['rect']):
                    # Sfx: Explosion
                    self._create_particles(pygame.Vector2(enemy['rect'].center), self.COLOR_ENEMY_1, 20, 5, 100)
                    self.player_projectiles.remove(proj)
                    self.enemies.remove(enemy)
                    self.score += 1
                    reward += 1
                    
                    # Chance to drop powerup
                    if self.rng.random() < 0.1: # 10% chance
                        ptype = self.rng.choice(['shield', 'firerate'])
                        self.powerups.append({
                            'rect': pygame.Rect(enemy['rect'].centerx - 10, enemy['rect'].centery - 10, 20, 20),
                            'type': ptype
                        })
                    break

        # Enemy projectiles vs base (or shielded player)
        for proj in self.enemy_projectiles[:]:
            collided = False
            # Check collision with shielded player first
            if self.player_powerup['type'] == 'shield':
                player_shield_rect = self.player.inflate(20, 20)
                if player_shield_rect.colliderect(proj):
                    # Sfx: Shield hit
                    self.enemy_projectiles.remove(proj)
                    self._create_particles(pygame.Vector2(proj.center), self.COLOR_POWERUP_SHIELD, 10, 2, 60)
                    collided = True
            
            # If not hit shield, check collision with base
            if not collided and self.base_rect.colliderect(proj):
                # Sfx: Base hit
                self.enemy_projectiles.remove(proj)
                self.base_health -= 5
                self._create_particles(pygame.Vector2(proj.centerx, self.HEIGHT - 25), self.COLOR_ENEMY_PROJ, 15, 3, 80)
                self.screen_shake = 15

        # Player vs powerups
        for powerup in self.powerups[:]:
            if self.player.colliderect(powerup['rect']):
                # Sfx: Powerup collect
                self.powerups.remove(powerup)
                self.player_powerup['type'] = powerup['type']
                self.player_powerup['duration'] = 7.0 # 7 seconds
                self.score += 5
                reward += 5
        
        # Assertions
        self.base_health = max(0, min(100, self.base_health))
        
        return reward

    def _get_observation(self):
        # Calculate screen offset for shake
        offset_x, offset_y = 0, 0
        if self.screen_shake > 0:
            offset_x = self.rng.integers(-1, 2) * self.screen_shake
            offset_y = self.rng.integers(-1, 2) * self.screen_shake

        # --- Render all game elements ---
        self.screen.fill(self.COLOR_BG)
        self._render_background(int(offset_x), int(offset_y))
        self._render_entities(int(offset_x), int(offset_y))
        self._render_particles(int(offset_x), int(offset_y))
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, ox, oy):
        # Stars
        for star in self.stars:
            pos = star['pos']
            pos.y = (pos.y + star['speed'] * (1/self.FPS)) % self.HEIGHT
            color_val = star['color']
            pygame.draw.rect(self.screen, (color_val, color_val, color_val), (pos.x + ox, pos.y + oy, star['size'], star['size']))

        # Planet surface
        planet_rect = pygame.Rect(ox, self.HEIGHT - 30 + oy, self.WIDTH, 60)
        pygame.draw.ellipse(self.screen, (30, 50, 80), planet_rect)
        # Base dome
        base_center = (self.WIDTH // 2 + ox, self.HEIGHT - 30 + oy)
        pygame.gfxdraw.filled_circle(self.screen, base_center[0], base_center[1], 40, (100, 120, 150))
        pygame.gfxdraw.aacircle(self.screen, base_center[0], base_center[1], 40, (150, 180, 220))

    def _render_entities(self, ox, oy):
        # Player
        player_draw_rect = self.player.move(ox, oy)
        pygame.gfxdraw.filled_trigon(self.screen, player_draw_rect.left, player_draw_rect.bottom, player_draw_rect.right, player_draw_rect.bottom, player_draw_rect.centerx, player_draw_rect.top, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_draw_rect.centerx, player_draw_rect.centery, 20, self.COLOR_PLAYER_GLOW)
        if self.player_powerup['type'] == 'shield':
            alpha = int(100 + 100 * math.sin(self.game_timer * 10)) # Pulsing effect
            shield_color = (self.COLOR_POWERUP_SHIELD[0], self.COLOR_POWERUP_SHIELD[1], self.COLOR_POWERUP_SHIELD[2], alpha)
            pygame.gfxdraw.filled_circle(self.screen, player_draw_rect.centerx, player_draw_rect.centery, 25, shield_color)
            pygame.gfxdraw.aacircle(self.screen, player_draw_rect.centerx, player_draw_rect.centery, 25, shield_color)

        # Player Projectiles
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, proj.move(ox, oy))
            pygame.gfxdraw.box(self.screen, proj.inflate(4,0).move(ox-2, oy), (*self.COLOR_PLAYER_PROJ, 50))

        # Enemies
        for enemy in self.enemies:
            e_rect = enemy['rect'].move(ox, oy)
            color = self.COLOR_ENEMY_1 if enemy['type'] == 1 else (self.COLOR_ENEMY_2 if enemy['type'] == 2 else self.COLOR_ENEMY_3)
            pygame.draw.rect(self.screen, color, e_rect)
            pygame.gfxdraw.aacircle(self.screen, e_rect.centerx, e_rect.centery, 20, self.COLOR_ENEMY_GLOW)

        # Enemy Projectiles
        for proj in self.enemy_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_PROJ, proj.move(ox, oy))

        # Powerups
        for powerup in self.powerups:
            p_rect = powerup['rect'].move(ox, oy)
            color = self.COLOR_POWERUP_SHIELD if powerup['type'] == 'shield' else self.COLOR_POWERUP_FIRERATE
            pygame.draw.rect(self.screen, color, p_rect)
            pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, p_rect, 2)
            letter = 'S' if powerup['type'] == 'shield' else 'F'
            text = self.font_small.render(letter, True, self.COLOR_UI_TEXT)
            self.screen.blit(text, text.get_rect(center=p_rect.center))
    
    def _render_particles(self, ox, oy):
        for p in self.particles:
            size = max(0, int(p['lifetime'] * p['size_mult']))
            if size > 0:
                pygame.draw.rect(self.screen, p['color'], (p['pos'][0] + ox, p['pos'][1] + oy, size, size))

    def _render_ui(self):
        # Health Bar
        health_ratio = self.base_health / 100
        bar_width = int(200 * health_ratio)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 10, 200, 20))
        if bar_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (10, 10, bar_width, 20))
        health_text = self.font_small.render(f"BASE HEALTH: {int(self.base_health)}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, score_text.get_rect(topright=(self.WIDTH - 10, 10)))

        # Timer
        time_left = max(0, self.GAME_DURATION_SECONDS - self.game_timer)
        minutes, seconds = divmod(int(time_left), 60)
        timer_text = self.font_small.render(f"TIME: {minutes:02}:{seconds:02}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, timer_text.get_rect(midtop=(self.WIDTH // 2, 10)))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        message = "MISSION SUCCESS" if self.base_health > 0 else "GAME OVER"
        text = self.font_large.render(message, True, self.COLOR_UI_TEXT)
        self.screen.blit(text, text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20)))

        score_text = self.font_small.render(f"FINAL SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, score_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 30)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "game_timer": self.game_timer
        }

    def _create_particles(self, pos, color, count, size_mult, speed):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.rng.uniform(speed/2, speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.rng.uniform(0.5, 1.2),
                'color': color,
                'size_mult': size_mult
            })

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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Planetary Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # The observation is a numpy array (H, W, C), Pygame needs a surface.
        # We transpose it to (W, H, C) for pygame.surfarray.make_surface.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()