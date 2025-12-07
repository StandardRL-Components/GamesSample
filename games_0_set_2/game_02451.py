
# Generated: 2025-08-28T04:53:20.436543
# Source Brief: brief_02451.md
# Brief Index: 2451

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to fire your weapon."
    )

    game_description = (
        "Pilot a powerful robot through three stages, blasting waves of enemies to achieve ultimate victory."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GROUND_Y = SCREEN_HEIGHT - 50

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GROUND = (40, 30, 30)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (50, 255, 150, 50)
    COLOR_ENEMY_BASIC = (255, 80, 80)
    COLOR_ENEMY_SINE = (255, 150, 50)
    COLOR_PLAYER_PROJ = (100, 200, 255)
    COLOR_ENEMY_PROJ = (255, 255, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR = (50, 200, 50)
    COLOR_HEALTH_BAR_BG = (100, 50, 50)
    COLOR_WHITE = (255, 255, 255)

    # Physics & Game Rules
    GRAVITY = 0.8
    PLAYER_SPEED = 7.0
    PLAYER_JUMP_POWER = -15.0
    PLAYER_MAX_HEALTH = 100
    PLAYER_SHOOT_COOLDOWN = 6  # frames
    PLAYER_PROJ_SPEED = 15

    ENEMY_MAX_HEALTH = 20
    ENEMY_PROJ_SPEED = 8
    
    MAX_STEPS = 1000 * 3 # A very high number to let timer/gameplay decide termination

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14)

        self.np_random = None
        self.game_over_message = ""
        
        self.reset()
        
        # This can be uncommented for self-validation during development
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.game_over_message = ""

        self.stage = 1
        self.stage_timer = 60 * self.FPS

        # Player state
        self.player_rect = pygame.Rect(100, self.GROUND_Y - 50, 30, 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.on_ground = False
        self.player_shoot_timer = 0
        self.player_last_move_dir = 1 # 1 for right, -1 for left

        # Entity lists
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.damage_flashes = []
        
        self._spawn_background()
        self._spawn_enemies_for_stage()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward_this_step = -0.01  # Small penalty for existing
        self.steps += 1
        self.stage_timer -= 1
        
        # --- Handle Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held)

        # --- Update Game State ---
        self._update_player()
        reward_this_step += self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        self._update_damage_flashes()
        self._update_background()

        # --- Handle Collisions ---
        reward_this_step += self._handle_collisions()

        # --- Check for Stage Progression ---
        if not self.game_over and not self.enemies:
            reward_this_step += self._advance_stage()

        # --- Check Termination Conditions ---
        terminated = False
        if not self.game_over:
            if self.player_health <= 0:
                self.game_over = True
                self.game_over_message = "ROBOT DESTROYED"
                reward_this_step -= 50
                # sfx: player_explosion
            elif self.stage_timer <= 0:
                self.game_over = True
                self.game_over_message = "TIME UP"
                reward_this_step -= 50
                # sfx: game_over_timer
            elif self.game_won:
                 self.game_over = True
                 self.game_over_message = "VICTORY!"
                 reward_this_step += 500
                 # sfx: victory_fanfare
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True # End episode if it runs too long
            if not self.game_over_message:
                self.game_over_message = "MAX STEPS REACHED"

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
            self.player_last_move_dir = -1
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
            self.player_last_move_dir = 1
        else:
            self.player_vel.x = 0

        # Jumping
        if movement == 1 and self.on_ground:  # Up
            self.player_vel.y = self.PLAYER_JUMP_POWER
            self.on_ground = False
            # sfx: player_jump

        # Shooting
        self.player_shoot_timer = max(0, self.player_shoot_timer - 1)
        if space_held and self.player_shoot_timer == 0:
            self.player_shoot_timer = self.PLAYER_SHOOT_COOLDOWN
            proj_start_pos = self.player_rect.center + pygame.Vector2(20 * self.player_last_move_dir, -10)
            self.player_projectiles.append(pygame.Rect(proj_start_pos.x, proj_start_pos.y, 12, 6))
            self._create_particles(proj_start_pos, 5, self.COLOR_PLAYER_PROJ, 2, 5, 0.5) # Muzzle flash
            # sfx: player_shoot

    def _update_player(self):
        # Apply physics
        self.player_vel.y += self.GRAVITY
        self.player_rect.x += self.player_vel.x
        self.player_rect.y += self.player_vel.y

        # World boundaries
        self.player_rect.left = max(0, self.player_rect.left)
        self.player_rect.right = min(self.SCREEN_WIDTH, self.player_rect.right)
        
        # Ground collision
        if self.player_rect.bottom >= self.GROUND_Y:
            self.player_rect.bottom = self.GROUND_Y
            self.player_vel.y = 0
            if not self.on_ground: # Landing
                self._create_particles(self.player_rect.midbottom, 5, self.COLOR_WHITE, 1, 3, 1) # Dust puff
                # sfx: player_land
            self.on_ground = True

    def _update_enemies(self):
        reward = 0
        base_speed = 1 + (self.stage - 1) * 0.5
        shoot_chance = 0.005 + (self.stage - 1) * 0.005

        for enemy in self.enemies:
            # Movement
            if enemy['type'] == 'basic':
                enemy['rect'].x += enemy['vel_x']
                if enemy['rect'].left < 0 or enemy['rect'].right > self.SCREEN_WIDTH:
                    enemy['vel_x'] *= -1
            elif enemy['type'] == 'sine':
                enemy['t'] += 0.05
                enemy['rect'].x += enemy['vel_x']
                enemy['rect'].y = enemy['start_y'] + math.sin(enemy['t'] * enemy['freq']) * enemy['amp']
                if enemy['rect'].left < 0 or enemy['rect'].right > self.SCREEN_WIDTH:
                    enemy['vel_x'] *= -1
            
            # Shooting
            if self.np_random.random() < shoot_chance:
                direction = pygame.Vector2(self.player_rect.center) - pygame.Vector2(enemy['rect'].center)
                if direction.length() > 0:
                    direction.normalize_ip()
                    proj_vel = direction * self.ENEMY_PROJ_SPEED
                    proj_rect = pygame.Rect(enemy['rect'].centerx - 4, enemy['rect'].centery - 4, 8, 8)
                    self.enemy_projectiles.append({'rect': proj_rect, 'vel': proj_vel})
                    # sfx: enemy_shoot
        return reward

    def _update_projectiles(self):
        # Player
        for proj in self.player_projectiles[:]:
            proj.x += self.player_last_move_dir * self.PLAYER_PROJ_SPEED
            self._create_particles(proj.center, 1, self.COLOR_PLAYER_PROJ, 1, 2, 0.2, (100, 150, 200, 100)) # Trail
            if proj.right < 0 or proj.left > self.SCREEN_WIDTH:
                self.player_projectiles.remove(proj)
        
        # Enemy
        for proj_data in self.enemy_projectiles[:]:
            proj_data['rect'].x += proj_data['vel'].x
            proj_data['rect'].y += proj_data['vel'].y
            self._create_particles(proj_data['rect'].center, 1, self.COLOR_ENEMY_PROJ, 1, 2, 0.2, (200, 200, 100, 100)) # Trail
            if not self.screen.get_rect().colliderect(proj_data['rect']):
                self.enemy_projectiles.remove(proj_data)

    def _handle_collisions(self):
        reward = 0
        # Player projectiles vs Enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if enemy['rect'].colliderect(proj):
                    enemy['health'] -= 10
                    reward += 0.1
                    self._create_particles(proj.center, 15, self.COLOR_ENEMY_BASIC, 2, 8) # Hit spark
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    # sfx: enemy_hit
                    if enemy['health'] <= 0:
                        self._create_particles(enemy['rect'].center, 50, enemy['color'], 3, 15, 2) # Explosion
                        self.enemies.remove(enemy)
                        self.score += 100
                        reward += 10
                        # sfx: enemy_explode
                    break
        
        # Enemy projectiles vs Player
        for proj_data in self.enemy_projectiles[:]:
            if self.player_rect.colliderect(proj_data['rect']):
                self.player_health -= 5
                self.player_health = max(0, self.player_health)
                self._create_particles(proj_data['rect'].center, 20, self.COLOR_PLAYER, 2, 8)
                self.damage_flashes.append({'alpha': 150, 'color': self.COLOR_ENEMY_BASIC})
                if proj_data in self.enemy_projectiles: self.enemy_projectiles.remove(proj_data)
                # sfx: player_hit
                break
        
        return reward

    def _advance_stage(self):
        self.stage += 1
        self.score += 500
        
        if self.stage > 3:
            self.game_won = True
            return 0 # Final victory reward is terminal
        else:
            self.stage_timer = 60 * self.FPS
            self._spawn_enemies_for_stage()
            self.player_projectiles.clear()
            self.enemy_projectiles.clear()
            # sfx: stage_clear
            return 100 # Stage clear bonus

    def _spawn_enemies_for_stage(self):
        self.enemies.clear()
        num_enemies = 5
        base_speed = 1.5 + (self.stage - 1) * 0.75
        
        for i in range(num_enemies):
            x = self.np_random.integers(self.SCREEN_WIDTH // 2, self.SCREEN_WIDTH - 50)
            y = self.np_random.integers(100, self.GROUND_Y - 80)
            
            enemy_type = 'basic'
            if self.stage > 1 and i % 2 != 0:
                enemy_type = 'sine'

            if enemy_type == 'basic':
                self.enemies.append({
                    'rect': pygame.Rect(x, y, 40, 40),
                    'health': self.ENEMY_MAX_HEALTH,
                    'vel_x': base_speed * (1 if self.np_random.random() > 0.5 else -1),
                    'type': 'basic',
                    'color': self.COLOR_ENEMY_BASIC
                })
            elif enemy_type == 'sine':
                 self.enemies.append({
                    'rect': pygame.Rect(x, y, 35, 35),
                    'health': self.ENEMY_MAX_HEALTH,
                    'vel_x': base_speed * 0.75 * (1 if self.np_random.random() > 0.5 else -1),
                    'type': 'sine',
                    'color': self.COLOR_ENEMY_SINE,
                    'start_y': y,
                    't': self.np_random.random() * math.pi * 2,
                    'freq': self.np_random.uniform(0.5, 1.5),
                    'amp': self.np_random.integers(20, 60),
                })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        for flash in self.damage_flashes:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((*flash['color'], flash['alpha']))
            self.screen.blit(flash_surface, (0, 0))

        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))
        pygame.draw.line(self.screen, (80, 70, 70), (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 2)
        
        # Parallax Stars
        for layer in self.bg_layers:
            for star in layer['stars']:
                star[0] = (star[0] - layer['speed']) % self.SCREEN_WIDTH
                pygame.gfxdraw.filled_circle(self.screen, int(star[0]), int(star[1]), int(star[2]), layer['color'])

    def _render_game(self):
        # Particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] -= p['shrink']
            if p['radius'] > 0:
                alpha = max(0, min(255, int(255 * (p['lifespan'] / p['start_life']))))
                color = (*p['color'][:3], alpha) if len(p['color']) == 4 else p['color']
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

        # Player Projectiles
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, proj)
            pygame.draw.rect(self.screen, self.COLOR_WHITE, proj.inflate(4, 4), 1)

        # Enemy Projectiles
        for proj_data in self.enemy_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, proj_data['rect'].centerx, proj_data['rect'].centery, proj_data['rect'].width // 2, self.COLOR_ENEMY_PROJ)
            pygame.gfxdraw.aacircle(self.screen, proj_data['rect'].centerx, proj_data['rect'].centery, proj_data['rect'].width // 2, self.COLOR_WHITE)

        # Enemies
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, enemy['color'], enemy['rect'])
            pygame.draw.rect(self.screen, self.COLOR_WHITE, enemy['rect'], 2)
            # Enemy health bar
            if enemy['health'] < self.ENEMY_MAX_HEALTH:
                health_pct = enemy['health'] / self.ENEMY_MAX_HEALTH
                bar_width = int(enemy['rect'].width * health_pct)
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (enemy['rect'].x, enemy['rect'].y - 8, bar_width, 4))

        # Player
        # Simple bobbing animation
        bob = math.sin(self.steps * 0.2) * 2 if self.on_ground and self.player_vel.x == 0 else 0
        player_draw_rect = self.player_rect.copy()
        player_draw_rect.y += bob
        
        # Glow effect
        glow_surface = pygame.Surface((self.player_rect.width * 2, self.player_rect.height * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surface, self.COLOR_PLAYER_GLOW, glow_surface.get_rect())
        self.screen.blit(glow_surface, (player_draw_rect.centerx - glow_surface.get_width()//2, player_draw_rect.centery - glow_surface.get_height()//2), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_draw_rect)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, player_draw_rect, 2)
        # Player "eye"
        eye_pos_x = player_draw_rect.centerx + 5 * self.player_last_move_dir
        eye_pos_y = player_draw_rect.centery - 10
        pygame.draw.rect(self.screen, self.COLOR_WHITE, (eye_pos_x - 3, eye_pos_y - 2, 6, 4))
        
    def _render_ui(self):
        # Health Bar
        health_pct = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_pct), 20))
        health_text = self.font_small.render(f"HP: {int(self.player_health)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 13))

        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Timer
        time_sec = self.stage_timer // self.FPS
        time_text = self.font_medium.render(f"TIME: {time_sec}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH // 2 - time_text.get_width() // 2, 10))

        # Stage
        stage_text = self.font_medium.render(f"STAGE: {self.stage}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (10, 40))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text_surface = self.font_large.render(self.game_over_message, True, self.COLOR_WHITE)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage}

    def _spawn_background(self):
        self.bg_layers = []
        for speed, num_stars, radius, color in [(0.2, 50, 1, (40, 40, 60)), (0.5, 30, 2, (70, 70, 90))]:
            stars = []
            for _ in range(num_stars):
                stars.append([self.np_random.random() * self.SCREEN_WIDTH, self.np_random.random() * self.GROUND_Y, radius])
            self.bg_layers.append({'stars': stars, 'speed': speed, 'color': color})
    
    def _update_background(self):
        # This is handled during rendering to avoid storing state for things that don't affect gameplay
        pass

    def _create_particles(self, pos, count, color, min_radius, max_radius, speed_mult=1.0, force_color=None):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            radius = self.np_random.uniform(min_radius, max_radius)
            lifespan = self.np_random.integers(10, 25)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'lifespan': lifespan,
                'start_life': lifespan,
                'radius': radius,
                'shrink': radius / lifespan,
                'color': force_color if force_color else color
            })
            
    def _update_particles(self):
        self.particles[:] = [p for p in self.particles if p['lifespan'] > 0]
        
    def _update_damage_flashes(self):
        for flash in self.damage_flashes[:]:
            flash['alpha'] -= 25
            if flash['alpha'] <= 0:
                self.damage_flashes.remove(flash)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        print("✓ Running implementation validation...")
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

if __name__ == "__main__":
    env = GameEnv()
    env.validate_implementation()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Robot Annihilator")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    while running:
        # Action defaults to no-op
        movement, space, shift = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_UP]:
            movement = 1

        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
        
        env.clock.tick(env.FPS)
        
    env.close()