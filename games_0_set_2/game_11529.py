import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
from collections import deque
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    TerraCore Defense: A Gymnasium environment where the player defends a central core.

    The player must survive for 1000 steps against waves of incoming enemies.
    They can shoot projectiles, place defensive terrain, and must manage their
    core's health, which is directly tied to its size.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) - Controls aiming reticle
    - actions[1]: Space button (0=released, 1=held) - Fires a projectile
    - actions[2]: Shift button (0=released, 1=held) - Places a terrain block

    Observation Space: Box(0, 255, (400, 640, 3), np.uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +0.1 for each enemy hit by a projectile.
    - -0.1 for the core being hit by an enemy.
    - +1.0 for destroying an enemy.
    - +100 for surviving the full 1000 steps.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Defend the central core from waves of incoming enemies by shooting projectiles and placing defensive terrain."
    user_guide = "Use the arrow keys (↑↓←→) to move the aiming reticle. Press space to fire a projectile and shift to place a terrain block."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1000
    FPS = 30  # Assumed frame rate for smooth visuals

    # Colors
    COLOR_BG = (15, 18, 23)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_ENEMY_SQUARE = (255, 50, 50)
    COLOR_ENEMY_TRIANGLE = (255, 120, 50)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_TERRAIN = (100, 110, 120)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BAR = (40, 200, 100)
    COLOR_UI_BAR_BG = (60, 60, 60)
    COLOR_RETICLE = (255, 255, 255)

    # Player Core
    CORE_POS = (WIDTH // 2, HEIGHT - 50)
    MAX_CORE_RADIUS = 30
    MIN_CORE_RADIUS = 10
    CORE_REGEN_RATE = 0.02  # Radius points per step of inactivity
    CORE_HIT_DAMAGE = 5  # Radius lost per hit

    # Aiming
    RETICLE_SPEED = 5
    MIN_LAUNCH_POWER = 5
    MAX_LAUNCH_POWER = 15

    # Projectiles
    PROJECTILE_COOLDOWN = 10  # steps
    PROJECTILE_RADIUS = 4
    PROJECTILE_GRAVITY = 0.1
    PROJECTILE_TRAIL_LENGTH = 8

    # Terrain
    TERRAIN_COOLDOWN = 30  # steps
    TERRAIN_SIZE = (40, 40)
    MAX_TERRAIN_BLOCKS = 8

    # Enemies
    INITIAL_SPAWN_RATE = 0.01
    SPAWN_RATE_INCREASE = 0.0015 / 100  # Per step
    INITIAL_ENEMY_SPEED = 1.0
    ENEMY_SPEED_INCREASE = 0.1 / 100 # Per step
    MAX_ENEMIES = 50

    # Particles
    PARTICLE_LIFESPAN = 20
    PARTICLES_PER_EXPLOSION = 25

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.core_radius = 0
        self.aim_reticle = (0, 0)
        self.launch_power = 0
        self.projectile_cooldown_timer = 0
        self.terrain_cooldown_timer = 0
        self.enemy_spawn_rate = 0
        self.enemy_speed_multiplier = 0
        self.projectiles = []
        self.enemies = []
        self.terrain_blocks = deque(maxlen=self.MAX_TERRAIN_BLOCKS)
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.core_radius = self.MAX_CORE_RADIUS
        self.aim_reticle = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.launch_power = (self.MIN_LAUNCH_POWER + self.MAX_LAUNCH_POWER) / 2

        self.projectile_cooldown_timer = 0
        self.terrain_cooldown_timer = 0

        self.enemy_spawn_rate = self.INITIAL_SPAWN_RATE
        self.enemy_speed_multiplier = self.INITIAL_ENEMY_SPEED

        self.projectiles = []
        self.enemies = []
        self.terrain_blocks.clear()
        self.particles = []

        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # --- Handle Input and Cooldowns ---
        action_taken = self._handle_input(action)
        self._update_cooldowns()

        # --- Update Game Logic ---
        if not action_taken:
            self._regenerate_core()

        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        self._spawn_enemies()

        # --- Handle Collisions ---
        reward += self._handle_collisions()

        # --- Update Difficulty ---
        self.enemy_spawn_rate += self.SPAWN_RATE_INCREASE
        self.enemy_speed_multiplier += self.ENEMY_SPEED_INCREASE

        # --- Check Termination Conditions ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Survived
            reward += 100.0

        return self._get_observation(), reward, terminated or truncated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        action_taken = False

        # 1. Aiming Reticle Movement
        if movement == 1: self.aim_reticle[1] -= self.RETICLE_SPEED  # Up
        elif movement == 2: self.aim_reticle[1] += self.RETICLE_SPEED  # Down
        
        # Clamp reticle vertically
        self.aim_reticle[1] = np.clip(self.aim_reticle[1], 20, self.CORE_POS[1] - 20)
        
        # Aiming power is based on horizontal distance from a vertical line through the core
        if movement == 3: self.aim_reticle[0] -= self.RETICLE_SPEED # Left
        elif movement == 4: self.aim_reticle[0] += self.RETICLE_SPEED # Right
        
        # Clamp reticle horizontally
        self.aim_reticle[0] = np.clip(self.aim_reticle[0], 20, self.WIDTH - 20)


        # 2. Fire Projectile (on space press)
        if space_held and not self.prev_space_held and self.projectile_cooldown_timer == 0:
            self._fire_projectile()
            action_taken = True
        self.prev_space_held = space_held

        # 3. Place Terrain (on shift press)
        if shift_held and not self.prev_shift_held and self.terrain_cooldown_timer == 0:
            self._place_terrain()
            action_taken = True
        self.prev_shift_held = shift_held

        return action_taken

    def _update_cooldowns(self):
        if self.projectile_cooldown_timer > 0:
            self.projectile_cooldown_timer -= 1
        if self.terrain_cooldown_timer > 0:
            self.terrain_cooldown_timer -= 1

    def _regenerate_core(self):
        self.core_radius = min(self.MAX_CORE_RADIUS, self.core_radius + self.CORE_REGEN_RATE)

    def _fire_projectile(self):
        self.projectile_cooldown_timer = self.PROJECTILE_COOLDOWN
        
        direction = self.aim_reticle - np.array(self.CORE_POS)
        distance = np.linalg.norm(direction)
        if distance == 0: return
        
        power_ratio = abs(self.aim_reticle[0] - self.CORE_POS[0]) / (self.WIDTH / 2)
        power = self.MIN_LAUNCH_POWER + power_ratio * (self.MAX_LAUNCH_POWER - self.MIN_LAUNCH_POWER)
        
        size_modifier = 1.0 + (self.MAX_CORE_RADIUS - self.core_radius) / self.MAX_CORE_RADIUS
        
        velocity = (direction / distance) * power * size_modifier
        
        self.projectiles.append({
            'pos': np.array(self.CORE_POS, dtype=float),
            'vel': velocity,
            'trail': deque(maxlen=self.PROJECTILE_TRAIL_LENGTH)
        })

    def _place_terrain(self):
        self.terrain_cooldown_timer = self.TERRAIN_COOLDOWN
        pos_x = self.aim_reticle[0] - self.TERRAIN_SIZE[0] / 2
        pos_y = self.aim_reticle[1] - self.TERRAIN_SIZE[1] / 2
        new_block = pygame.Rect(pos_x, pos_y, self.TERRAIN_SIZE[0], self.TERRAIN_SIZE[1])
        self.terrain_blocks.append(new_block)

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p['trail'].append(p['pos'].copy())
            p['pos'] += p['vel']
            p['vel'][1] += self.PROJECTILE_GRAVITY
            if not (0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT):
                self.projectiles.remove(p)

    def _update_enemies(self):
        for e in self.enemies:
            direction = np.array(self.CORE_POS) - e['pos']
            dist = np.linalg.norm(direction)
            if dist > 1:
                direction /= dist

            velocity = direction * e['speed'] * self.enemy_speed_multiplier
            
            next_pos_rect = pygame.Rect(e['pos'][0] - e['size']/2 + velocity[0], e['pos'][1] - e['size']/2 + velocity[1], e['size'], e['size'])
            for block in self.terrain_blocks:
                if block.colliderect(next_pos_rect):
                    if abs(velocity[0]) > abs(velocity[1]):
                        velocity[0] = 0
                    else:
                        velocity[1] = 0
                    break

            e['pos'] += velocity

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _spawn_enemies(self):
        if self.np_random.random() < self.enemy_spawn_rate and len(self.enemies) < self.MAX_ENEMIES:
            x_pos = self.np_random.uniform(20, self.WIDTH - 20)
            enemy_type = self.np_random.choice(['square', 'triangle'])
            
            if enemy_type == 'square':
                self.enemies.append({
                    'pos': np.array([x_pos, 0], dtype=float),
                    'type': 'square', 'size': 18, 'health': 2, 'speed': 1.0
                })
            else: # triangle
                self.enemies.append({
                    'pos': np.array([x_pos, 0], dtype=float),
                    'type': 'triangle', 'size': 20, 'health': 1, 'speed': 1.5
                })

    def _handle_collisions(self):
        reward = 0

        for p in self.projectiles[:]:
            proj_rect = pygame.Rect(p['pos'][0]-self.PROJECTILE_RADIUS, p['pos'][1]-self.PROJECTILE_RADIUS, self.PROJECTILE_RADIUS*2, self.PROJECTILE_RADIUS*2)
            for e in self.enemies[:]:
                enemy_rect = pygame.Rect(e['pos'][0]-e['size']/2, e['pos'][1]-e['size']/2, e['size'], e['size'])
                if enemy_rect.colliderect(proj_rect):
                    reward += 0.1
                    e['health'] -= 1
                    if e['health'] <= 0:
                        self._create_explosion(e['pos'], e['type'])
                        self.enemies.remove(e)
                        self.score += 1
                        reward += 1.0
                    if p in self.projectiles:
                        self.projectiles.remove(p)
                    break

        core_rect = pygame.Rect(self.CORE_POS[0]-self.core_radius, self.CORE_POS[1]-self.core_radius, self.core_radius*2, self.core_radius*2)
        for e in self.enemies[:]:
            enemy_rect = pygame.Rect(e['pos'][0]-e['size']/2, e['pos'][1]-e['size']/2, e['size'], e['size'])
            if enemy_rect.colliderect(core_rect):
                self._create_explosion(e['pos'], e['type'])
                self.enemies.remove(e)
                self.core_radius -= self.CORE_HIT_DAMAGE
                reward -= 0.1
                if self.core_radius < self.MIN_CORE_RADIUS:
                    self.core_radius = self.MIN_CORE_RADIUS
                    self.game_over = True
        
        return reward

    def _create_explosion(self, pos, enemy_type):
        color = self.COLOR_ENEMY_SQUARE if enemy_type == 'square' else self.COLOR_ENEMY_TRIANGLE
        for _ in range(self.PARTICLES_PER_EXPLOSION):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(10, self.PARTICLE_LIFESPAN),
                'color': color,
                'size': self.np_random.uniform(1, 4)
            })

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over or self.steps >= self.MAX_STEPS:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for block in self.terrain_blocks:
            pygame.draw.rect(self.screen, self.COLOR_TERRAIN, block)

        for p in self.projectiles:
            for i, pos in enumerate(p['trail']):
                alpha = int(255 * (i / self.PROJECTILE_TRAIL_LENGTH))
                color = (*self.COLOR_PROJECTILE, alpha)
                temp_surf = pygame.Surface((self.PROJECTILE_RADIUS*2, self.PROJECTILE_RADIUS*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (self.PROJECTILE_RADIUS, self.PROJECTILE_RADIUS), self.PROJECTILE_RADIUS * (i/len(p['trail'])))
                self.screen.blit(temp_surf, (int(pos[0]-self.PROJECTILE_RADIUS), int(pos[1]-self.PROJECTILE_RADIUS)))

        for e in self.enemies:
            pos_int = (int(e['pos'][0]), int(e['pos'][1]))
            color = self.COLOR_ENEMY_SQUARE if e['type'] == 'square' else self.COLOR_ENEMY_TRIANGLE
            if e['type'] == 'square':
                size = int(e['size'])
                pygame.draw.rect(self.screen, color, (pos_int[0] - size//2, pos_int[1] - size//2, size, size))
            else:
                s = e['size']
                points = [
                    (pos_int[0], pos_int[1] - s * 0.58),
                    (pos_int[0] - s/2, pos_int[1] + s * 0.29),
                    (pos_int[0] + s/2, pos_int[1] + s * 0.29)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        core_pos_int = (int(self.CORE_POS[0]), int(self.CORE_POS[1]))
        radius = int(self.core_radius)
        for i in range(radius // 2, 0, -2):
            alpha = 40 - (i / (radius//2)) * 40
            pygame.gfxdraw.filled_circle(self.screen, core_pos_int[0], core_pos_int[1], radius + i, (*self.COLOR_PLAYER_GLOW, alpha))
        pygame.gfxdraw.aacircle(self.screen, core_pos_int[0], core_pos_int[1], radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, core_pos_int[0], core_pos_int[1], radius, self.COLOR_PLAYER)

        for p in self.projectiles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)

        for p in self.particles:
            alpha = max(0, int(255 * (p['lifespan'] / self.PARTICLE_LIFESPAN)))
            color = (*p['color'], alpha)
            size = int(p['size'])
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0,0,size,size))
            self.screen.blit(temp_surf, (int(p['pos'][0]-size/2), int(p['pos'][1]-size/2)))

        ret_pos_int = (int(self.aim_reticle[0]), int(self.aim_reticle[1]))
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (ret_pos_int[0]-5, ret_pos_int[1]), (ret_pos_int[0]+5, ret_pos_int[1]), 1)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (ret_pos_int[0], ret_pos_int[1]-5), (ret_pos_int[0], ret_pos_int[1]+5), 1)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        steps_text = self.font_ui.render(f"TIMER: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        health_ratio = (self.core_radius - self.MIN_CORE_RADIUS) / (self.MAX_CORE_RADIUS - self.MIN_CORE_RADIUS)
        bar_w, bar_h = 20, 150
        bar_x, bar_y = 15, self.HEIGHT - bar_h - 15
        fill_h = bar_h * health_ratio
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y + (bar_h - fill_h), bar_w, fill_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_w, bar_h), 2)
        
        if self.projectile_cooldown_timer > 0:
            ratio = self.projectile_cooldown_timer / self.PROJECTILE_COOLDOWN
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, self.CORE_POS, int(self.core_radius * ratio), 1)
        if self.terrain_cooldown_timer > 0:
            ratio = self.terrain_cooldown_timer / self.TERRAIN_COOLDOWN
            pygame.draw.circle(self.screen, self.COLOR_TERRAIN, (int(self.aim_reticle[0]), int(self.aim_reticle[1])), int(15 * ratio), 1)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        if self.steps >= self.MAX_STEPS:
            text = "VICTORY"
            color = (100, 255, 100)
        else:
            text = "CORE DESTROYED"
            color = (255, 100, 100)
        
        game_over_surf = self.font_game_over.render(text, True, color)
        pos_x = self.WIDTH/2 - game_over_surf.get_width()/2
        pos_y = self.HEIGHT/2 - game_over_surf.get_height()/2
        self.screen.blit(game_over_surf, (pos_x, pos_y))


    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not work with the "dummy" video driver, so we unset it.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("TerraCore Defense")
    clock = pygame.time.Clock()
    
    terminated, truncated = False, False
    
    while not (terminated or truncated):
        # --- Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated, truncated = False, False

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()