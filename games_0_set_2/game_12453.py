import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:39:05.711742
# Source Brief: brief_02453.md
# Brief Index: 2453
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A cyberpunk gladiator battles in a gravity-flipping arena.

    This Gymnasium environment features a top-down arena combat game with a
    unique gravity-flipping mechanic. The agent controls a cyber-gladiator,
    fighting against waves of enemies. The visual design is a key focus,
    with neon aesthetics, particle effects, and smooth animations to create
    an engaging and high-quality gameplay experience.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0:None, 1:Up, 2:Down, 3:Left, 4:Right)
    - action[1]: Primary Attack (0:Released, 1:Held/Pressed)
    - action[2]: Flip Gravity (0:Released, 1:Held/Pressed)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Rewards:
    - +100 for clearing a round.
    - -100 for player death.
    - +1 for defeating an enemy.
    - +0.1 for landing a hit on an enemy.
    - -0.1 for taking damage.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Control a cyber-gladiator in a fast-paced, top-down arena shooter with a unique gravity-flipping mechanic."
    user_guide = "Use the arrow keys (↑↓←→) to move. Press space to shoot and shift to flip gravity."
    auto_advance = True


    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_GRID = (30, 20, 50)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_ENEMY = (255, 100, 0)
    COLOR_ENEMY_GLOW = (200, 80, 0)
    COLOR_PROJECTILE = (200, 255, 255)
    COLOR_TEXT = (220, 220, 255)
    COLOR_HEALTH_GREEN = (0, 255, 100)
    COLOR_HEALTH_RED = (255, 50, 50)
    COLOR_HEALTH_BG = (50, 50, 50)

    # Player
    PLAYER_SIZE = 12
    PLAYER_ACCELERATION = 1.0
    PLAYER_FRICTION = 0.90
    MAX_PLAYER_HEALTH = 100
    PLAYER_ATTACK_COOLDOWN = 200  # ms

    # Enemy
    ENEMY_SIZE = 10
    ENEMY_BASE_SPEED = 0.2
    ENEMY_BASE_HEALTH = 30
    ENEMY_CONTACT_DAMAGE = 10
    ENEMY_DAMAGE_COOLDOWN = 500 # ms

    # Gravity
    GRAVITY_STRENGTH = 0.4
    GRAVITY_FLIP_COOLDOWN = 2000  # ms

    # Projectile
    PROJECTILE_SPEED = 12
    PROJECTILE_SIZE = 4
    PROJECTILE_DAMAGE = 10
    PROJECTILE_LIFESPAN = 50 # steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)

        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize all state variables to prevent errors
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = 0
        self.last_player_damage_time = 0
        
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.gravity_direction = 1
        self.last_gravity_flip_time = 0
        self.gravity_flip_effect_timer = 0
        
        self.round_number = 1
        self.round_cleared_message_timer = 0
        
        self.last_attack_time = 0
        self.screen_shake_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = self.MAX_PLAYER_HEALTH
        self.last_player_damage_time = 0

        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.gravity_direction = 1
        self.last_gravity_flip_time = 0
        self.gravity_flip_effect_timer = 0
        
        self.round_number = 1
        self.round_cleared_message_timer = 0
        
        self._spawn_round()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        current_time = pygame.time.get_ticks()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Actions ---
        if space_held and current_time - self.last_attack_time > self.PLAYER_ATTACK_COOLDOWN:
            self._fire_projectile()
            self.last_attack_time = current_time
            # sfx: PlayerShoot.wav

        if shift_held and current_time - self.last_gravity_flip_time > self.GRAVITY_FLIP_COOLDOWN:
            self._flip_gravity()
            self.last_gravity_flip_time = current_time
            # sfx: GravityFlip.wav

        # --- Update Game State ---
        self._update_player(movement)
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        
        # --- Handle Collisions & Rewards ---
        collision_rewards = self._handle_collisions(current_time)
        reward += collision_rewards

        # --- Check for Round Clear ---
        if not self.enemies and not self.game_over:
            reward += 100
            self.score += 50 * self.round_number
            self.round_number += 1
            self.round_cleared_message_timer = self.FPS * 2 # 2 seconds
            self._spawn_round()
            # sfx: RoundClear.wav

        # --- Update Timers ---
        self.steps += 1
        if self.screen_shake_timer > 0: self.screen_shake_timer -= 1
        if self.round_cleared_message_timer > 0: self.round_cleared_message_timer -= 1
        if self.gravity_flip_effect_timer > 0: self.gravity_flip_effect_timer -= 1
            
        # --- Check Termination ---
        terminated = self.player_health <= 0
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            self.game_over = True
            reward -= 100
            self._create_explosion(self.player_pos, self.COLOR_PLAYER, 100)
            # sfx: PlayerDeath.wav
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Update Logic ---
    def _update_player(self, movement):
        acc = pygame.Vector2(0, 0)
        if movement == 1: acc.y = -self.PLAYER_ACCELERATION
        elif movement == 2: acc.y = self.PLAYER_ACCELERATION
        elif movement == 3: acc.x = -self.PLAYER_ACCELERATION
        elif movement == 4: acc.x = self.PLAYER_ACCELERATION
        
        self.player_vel += acc
        self.player_vel.y += self.gravity_direction * self.GRAVITY_STRENGTH
        self.player_vel *= self.PLAYER_FRICTION
        if self.player_vel.length() > 15: self.player_vel.scale_to_length(15)
        
        self.player_pos += self.player_vel
        
        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _update_enemies(self):
        for enemy in self.enemies:
            direction_to_player = self.player_pos - enemy['pos']
            if direction_to_player.length() > 0:
                direction_to_player.normalize_ip()
            
            enemy['vel'] += direction_to_player * enemy['speed']
            enemy['vel'].y += self.gravity_direction * self.GRAVITY_STRENGTH
            enemy['vel'] *= self.PLAYER_FRICTION
            if enemy['vel'].length() > 10: enemy['vel'].scale_to_length(10)
            
            enemy['pos'] += enemy['vel']
            
            enemy['pos'].x = np.clip(enemy['pos'].x, self.ENEMY_SIZE, self.WIDTH - self.ENEMY_SIZE)
            enemy['pos'].y = np.clip(enemy['pos'].y, self.ENEMY_SIZE, self.HEIGHT - self.ENEMY_SIZE)

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0 or not (0 < p['pos'].x < self.WIDTH and 0 < p['pos'].y < self.HEIGHT):
                self.projectiles.remove(p)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self, current_time):
        reward = 0
        
        # Projectiles vs Enemies
        for p in self.projectiles[:]:
            for enemy in self.enemies[:]:
                if p['pos'].distance_to(enemy['pos']) < self.ENEMY_SIZE + self.PROJECTILE_SIZE:
                    enemy['health'] -= self.PROJECTILE_DAMAGE
                    reward += 0.1
                    self.score += 1
                    self._create_explosion(p['pos'], self.COLOR_ENEMY, 10, 2)
                    # sfx: EnemyHit.wav
                    if p in self.projectiles: self.projectiles.remove(p)

                    if enemy['health'] <= 0:
                        reward += 1
                        self.score += 10
                        self._create_explosion(enemy['pos'], self.COLOR_ENEMY, 40)
                        # sfx: EnemyExplode.wav
                        if enemy in self.enemies: self.enemies.remove(enemy)
                    break
        
        # Player vs Enemies
        if current_time - self.last_player_damage_time > self.ENEMY_DAMAGE_COOLDOWN:
            for enemy in self.enemies:
                if self.player_pos.distance_to(enemy['pos']) < self.PLAYER_SIZE + self.ENEMY_SIZE:
                    self.player_health -= self.ENEMY_CONTACT_DAMAGE
                    self.player_health = max(0, self.player_health)
                    reward -= 0.1
                    self.last_player_damage_time = current_time
                    self.screen_shake_timer = 10
                    self._create_explosion(self.player_pos, self.COLOR_PLAYER, 20, 3)
                    # sfx: PlayerHit.wav
                    break
        return reward

    # --- Action Implementations ---
    def _fire_projectile(self):
        mouse_pos = self._get_aim_target()
        direction = (mouse_pos - self.player_pos).normalize()
        vel = direction * self.PROJECTILE_SPEED
        pos = self.player_pos + direction * (self.PLAYER_SIZE + 5)
        
        self.projectiles.append({
            'pos': pos, 'vel': vel, 'lifespan': self.PROJECTILE_LIFESPAN
        })
        # Muzzle flash
        for _ in range(5):
            lifespan = random.randint(5, 10)
            vel_particle = direction.rotate(random.uniform(-45, 45)) * random.uniform(2, 4)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel_particle, 'lifespan': lifespan, 'max_lifespan': lifespan,
                'color': self.COLOR_PROJECTILE, 'size': random.randint(1, 3)
            })

    def _flip_gravity(self):
        self.gravity_direction *= -1
        self.gravity_flip_effect_timer = self.FPS // 2 # 0.5 seconds
        for _ in range(150):
            lifespan = random.randint(20, 40)
            angle = random.uniform(0, 360)
            speed = random.uniform(2, 8)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            self.particles.append({
                'pos': pygame.Vector2(self.WIDTH/2, self.HEIGHT/2),
                'vel': vel, 'lifespan': lifespan, 'max_lifespan': lifespan,
                'color': random.choice([self.COLOR_PLAYER, self.COLOR_GRID, (100,100,255)]),
                'size': random.randint(1, 3)
            })

    # --- Game Logic Helpers ---
    def _get_aim_target(self):
        # In a real game, this would be the mouse cursor.
        # For an RL agent, we'll make it aim at the nearest enemy.
        if not self.enemies:
            return self.player_pos + pygame.Vector2(1,0) # Default aim right
        
        closest_enemy = min(self.enemies, key=lambda e: self.player_pos.distance_squared_to(e['pos']))
        return closest_enemy['pos']

    def _spawn_round(self):
        self.enemies.clear()
        num_enemies = 1 + self.round_number // 2
        
        # Difficulty scaling
        difficulty_mod = 1 + (self.round_number - 1) // 5
        enemy_speed = self.ENEMY_BASE_SPEED * (1 + 0.1 * difficulty_mod)
        enemy_health = self.ENEMY_BASE_HEALTH * (1 + 0.2 * difficulty_mod)
        
        for _ in range(num_enemies):
            while True:
                pos = pygame.Vector2(random.uniform(50, self.WIDTH-50), random.uniform(50, self.HEIGHT-50))
                if pos.distance_to(self.player_pos) > 150: # Spawn away from player
                    break
            
            self.enemies.append({
                'pos': pos, 'vel': pygame.Vector2(0, 0),
                'health': enemy_health, 'max_health': enemy_health,
                'speed': enemy_speed,
            })
            
    def _create_explosion(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            lifespan = random.randint(15, 30)
            angle = random.uniform(0, 360)
            speed = random.uniform(1, 5) * speed_mult
            vel = pygame.Vector2(speed, 0).rotate(angle)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'max_lifespan': lifespan,
                'color': color, 'size': random.uniform(1, 4)
            })

    # --- Rendering ---
    def _get_observation(self):
        render_surface = self.screen
        
        # Handle screen shake
        if self.screen_shake_timer > 0:
            shake_offset = (random.randint(-4, 4), random.randint(-4, 4))
            temp_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
            self._render_all(temp_surface)
            render_surface.blit(temp_surface, shake_offset)
        else:
            self._render_all(render_surface)

        arr = pygame.surfarray.array3d(render_surface)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
        
    def _render_all(self, surface):
        surface.fill(self.COLOR_BG)
        self._render_background(surface)
        self._render_particles(surface)
        self._render_projectiles(surface)
        if not self.game_over:
            self._render_player(surface)
        self._render_enemies(surface)
        self._render_ui(surface)
    
    def _render_background(self, surface):
        # Grid lines
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(surface, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(surface, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)
            
        # Gravity indicator arrows
        arrow_color = self.COLOR_PLAYER if self.gravity_flip_effect_timer > 0 else self.COLOR_GRID
        arrow_size = 10
        for y in [30, self.HEIGHT-30]:
            for x in range(40, self.WIDTH, 80):
                if self.gravity_direction == 1: # Down
                    pygame.draw.polygon(surface, arrow_color, [(x, y), (x-arrow_size/2, y-arrow_size), (x+arrow_size/2, y-arrow_size)])
                else: # Up
                    pygame.draw.polygon(surface, arrow_color, [(x, y), (x-arrow_size/2, y+arrow_size), (x+arrow_size/2, y+arrow_size)])

    def _render_glow_shape(self, surface, pos, radius, color, glow_color):
        # Draw multiple layers of circles for a neon glow effect
        for i in range(int(radius * 1.5), int(radius), -2):
            alpha = 40 * (1 - (i - radius) / (radius * 0.5))
            pygame.gfxdraw.filled_circle(surface, int(pos.x), int(pos.y), i, (*glow_color, alpha))
        pygame.gfxdraw.aacircle(surface, int(pos.x), int(pos.y), int(radius), color)
        pygame.gfxdraw.filled_circle(surface, int(pos.x), int(pos.y), int(radius), color)

    def _render_player(self, surface):
        self._render_glow_shape(surface, self.player_pos, self.PLAYER_SIZE, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
        self._render_health_bar(surface, self.player_pos, self.player_health, self.MAX_PLAYER_HEALTH, self.PLAYER_SIZE + 5)

    def _render_enemies(self, surface):
        for enemy in self.enemies:
            self._render_glow_shape(surface, enemy['pos'], self.ENEMY_SIZE, self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW)
            self._render_health_bar(surface, enemy['pos'], enemy['health'], enemy['max_health'], self.ENEMY_SIZE + 5)
            
    def _render_projectiles(self, surface):
        for p in self.projectiles:
            start_pos = p['pos']
            end_pos = p['pos'] - p['vel'] * 0.5
            pygame.draw.line(surface, self.COLOR_PROJECTILE, (int(start_pos.x), int(start_pos.y)), (int(end_pos.x), int(end_pos.y)), 3)
            pygame.gfxdraw.filled_circle(surface, int(p['pos'].x), int(p['pos'].y), self.PROJECTILE_SIZE, self.COLOR_PROJECTILE)

    def _render_particles(self, surface):
        for p in self.particles:
            alpha = 255 * (p['lifespan'] / p['max_lifespan']) if 'max_lifespan' in p else 255
            color_with_alpha = (*p['color'], alpha)
            try:
                pygame.gfxdraw.filled_circle(surface, int(p['pos'].x), int(p['pos'].y), int(p['size']), color_with_alpha)
            except TypeError: # Handle cases where color might not have alpha
                 pygame.gfxdraw.filled_circle(surface, int(p['pos'].x), int(p['pos'].y), int(p['size']), p['color'])


    def _render_health_bar(self, surface, pos, current_hp, max_hp, y_offset):
        bar_width = 30
        bar_height = 5
        x = pos.x - bar_width / 2
        y = pos.y - y_offset - bar_height
        
        health_ratio = max(0, current_hp / max_hp)
        
        pygame.draw.rect(surface, self.COLOR_HEALTH_BG, (x, y, bar_width, bar_height))
        pygame.draw.rect(surface, self.COLOR_HEALTH_GREEN, (x, y, bar_width * health_ratio, bar_height))

    def _render_ui(self, surface):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        surface.blit(score_text, (10, 10))
        
        round_text = self.font_small.render(f"ROUND: {self.round_number}", True, self.COLOR_TEXT)
        surface.blit(round_text, (self.WIDTH - round_text.get_width() - 10, 10))

        if self.round_cleared_message_timer > 0:
            alpha = min(255, self.round_cleared_message_timer * 5)
            clear_text = self.font_large.render("ROUND CLEARED", True, self.COLOR_TEXT)
            clear_text.set_alpha(alpha)
            text_rect = clear_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            surface.blit(clear_text, text_rect)
            
        if self.game_over:
            lose_text = self.font_large.render("DEFEATED", True, self.COLOR_HEALTH_RED)
            text_rect = lose_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            surface.blit(lose_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "round": self.round_number}

    def close(self):
        pygame.quit()


# --- Example Usage ---
if __name__ == "__main__":
    # The following code is for human testing and visualization.
    # It requires a display environment and will not run in a strictly headless setup.
    # To run, you might need to unset the SDL_VIDEODRIVER variable, e.g., by commenting out
    # the os.environ.setdefault line at the top of the file.
    
    try:
        env = GameEnv()
        obs, info = env.reset()
        
        pygame.display.set_caption("Cyberpunk Gladiator")
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            movement_action = 0 # None
            space_action = 0 # Released
            shift_action = 0 # Released
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement_action = 1
            elif keys[pygame.K_DOWN]: movement_action = 2
            elif keys[pygame.K_LEFT]: movement_action = 3
            elif keys[pygame.K_RIGHT]: movement_action = 4
            
            if keys[pygame.K_SPACE]: space_action = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

            action = [movement_action, space_action, shift_action]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Display the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}, Steps: {info['steps']}")
                obs, info = env.reset()
                total_reward = 0

            clock.tick(env.FPS)
            
        env.close()
    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("This example requires a display. It might not work in a headless environment.")
        print("Try commenting out 'os.environ.setdefault(\"SDL_VIDEODRIVER\", \"dummy\")' to run.")