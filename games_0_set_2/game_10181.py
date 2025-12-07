import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:58:02.555727
# Source Brief: brief_00181.md
# Brief Index: 181
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Navigate a celestial ship through a dangerous nebula. Flip gravity to dodge enemies and collect stardust to power your shield."
    )
    user_guide = (
        "Controls: Press Shift to flip gravity and Space to activate your shield. Collect yellow stardust to recharge."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_PLAYER = (0, 191, 255) # Deep Sky Blue
    COLOR_PLAYER_GLOW = (0, 191, 255, 50)
    COLOR_STARDUST = (255, 255, 0) # Yellow
    COLOR_STARDUST_GLOW = (255, 255, 0, 60)
    COLOR_SHIELD = (0, 255, 255) # Cyan
    COLOR_SHIELD_GLOW = (0, 255, 255, 100)
    COLOR_ENEMY = (255, 69, 0) # OrangeRed
    COLOR_ENEMY_GLOW = (255, 69, 0, 70)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_SHIELD_BAR = (0, 255, 255)
    COLOR_UI_SHIELD_BAR_BG = (50, 50, 80)
    
    # Player
    PLAYER_START_POS = pygame.Vector2(100, SCREEN_HEIGHT / 2)
    PLAYER_RADIUS = 12
    GRAVITY_ACCEL = 0.4
    PLAYER_MAX_VEL = 8
    
    # Shield
    SHIELD_COST = 30.0
    SHIELD_DURATION = 60 # steps (2 seconds at 30fps)
    MAX_SHIELD_ENERGY = 100.0
    
    # Stardust
    STARDUST_COUNT = 20
    STARDUST_RADIUS = 5
    STARDUST_REWARD = 1.0
    STARDUST_ENERGY_GAIN = 15.0
    
    # Enemies
    INITIAL_ENEMY_SPEED = 2.0
    ENEMY_SPEED_INCREASE = 0.05
    ENEMY_RADIUS = 15
    
    # Game
    MAX_STEPS = 10000
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        
        self.render_mode = render_mode
        self._background_stars = self._create_stars(200)

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.gravity_direction = None
        self.shield_energy = None
        self.shield_active_timer = None
        self.stardust = None
        self.enemies = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.enemy_speed = None
        self.enemy_spawn_timer = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.enemy_types = [
            {'amp': 50, 'freq': 0.02, 'offset': 0},      # Gentle Sine
            {'amp': 100, 'freq': 0.04, 'offset': math.pi}, # Wide Sine
            {'amp': 20, 'freq': 0.1, 'offset': 0},       # Fast Wobble
            {'amp': 80, 'freq': 0.01, 'offset': math.pi/2}, # Slow Drift
        ]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = self.PLAYER_START_POS.copy()
        self.player_vel = pygame.Vector2(0, 0)
        self.gravity_direction = 1 # 1 for down, -1 for up
        
        self.shield_energy = 25.0
        self.shield_active_timer = 0
        
        self.stardust = self._generate_stardust()
        self.enemies = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.enemy_speed = self.INITIAL_ENEMY_SPEED
        self.enemy_spawn_timer = 0
        
        self.prev_space_held = 0
        self.prev_shift_held = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.01 # Survival reward
        
        # --- 1. Handle Input & Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        if shift_pressed:
            self.gravity_direction *= -1
            # Sound: Gravity shift whoosh
            self._create_particles(self.player_pos, self.COLOR_PLAYER, 10, 2) # Flip effect

        if space_pressed and self.shield_energy >= self.SHIELD_COST and self.shield_active_timer <= 0:
            self.shield_energy -= self.SHIELD_COST
            self.shield_active_timer = self.SHIELD_DURATION
            # Sound: Shield activation hum
            
        # --- 2. Update Game State ---
        self.steps += 1
        if self.shield_active_timer > 0:
            self.shield_active_timer -= 1

        # Update player
        self.player_vel.y += self.gravity_direction * self.GRAVITY_ACCEL
        self.player_vel.y = np.clip(self.player_vel.y, -self.PLAYER_MAX_VEL, self.PLAYER_MAX_VEL)
        self.player_pos += self.player_vel
        
        # Player boundaries
        if self.player_pos.y - self.PLAYER_RADIUS < 0:
            self.player_pos.y = self.PLAYER_RADIUS
            self.player_vel.y *= -0.5 # Bounce
        if self.player_pos.y + self.PLAYER_RADIUS > self.SCREEN_HEIGHT:
            self.player_pos.y = self.SCREEN_HEIGHT - self.PLAYER_RADIUS
            self.player_vel.y *= -0.5 # Bounce

        # Update enemies
        self._update_enemies()
        
        # Update particles
        self._update_particles()
        
        # --- 3. Handle Collisions ---
        # Stardust collection
        collected_stardust = []
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_RADIUS, self.player_pos.y - self.PLAYER_RADIUS, self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2)
        for dust in self.stardust:
            if player_rect.colliderect(dust['rect']):
                collected_stardust.append(dust)
                self.score += self.STARDUST_REWARD
                reward += self.STARDUST_REWARD
                self.shield_energy = min(self.MAX_SHIELD_ENERGY, self.shield_energy + self.STARDUST_ENERGY_GAIN)
                # Sound: Stardust collection ping
                self._create_particles(pygame.Vector2(dust['rect'].center), self.COLOR_STARDUST, 15, 3, life=20)
        self.stardust = [d for d in self.stardust if d not in collected_stardust]

        # Enemy collision
        is_shielded = self.shield_active_timer > 0
        for enemy in self.enemies:
            distance = self.player_pos.distance_to(enemy['pos'])
            if distance < self.PLAYER_RADIUS + self.ENEMY_RADIUS:
                if is_shielded:
                    # Shield hit, destroy enemy
                    self.enemies.remove(enemy)
                    self.score += 5 # Bonus for shielded block
                    reward += 5
                    self.shield_active_timer = 0 # Shield breaks on impact
                    # Sound: Shield impact crackle
                    self._create_particles(enemy['pos'], self.COLOR_SHIELD, 30, 5)
                else:
                    # Game Over
                    self.game_over = True
                    reward = -100.0
                    # Sound: Player explosion
                    self._create_particles(self.player_pos, self.COLOR_PLAYER, 50, 7)
                    self._create_particles(self.player_pos, (255,255,255), 20, 3)
                    break
        
        # --- 4. Update Progression & Termination ---
        if self.steps % 100 == 0 and self.steps > 0:
            self.enemy_speed += self.ENEMY_SPEED_INCREASE
            
        terminated = self.game_over
        truncated = False
        if self.steps >= self.MAX_STEPS:
            terminated = True
            reward += 100.0 # Victory reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_enemies(self):
        # Spawn new enemies
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            spawn_y = self.np_random.uniform(self.ENEMY_RADIUS, self.SCREEN_HEIGHT - self.ENEMY_RADIUS)
            
            # Introduce new enemy types over time
            type_index = min(len(self.enemy_types) - 1, self.steps // 500)
            enemy_type = self.enemy_types[type_index]

            self.enemies.append({
                'pos': pygame.Vector2(self.SCREEN_WIDTH + self.ENEMY_RADIUS, spawn_y),
                'base_y': spawn_y,
                'type': enemy_type,
                'time': 0
            })
            self.enemy_spawn_timer = self.np_random.integers(60, 120)

        # Move existing enemies
        for enemy in self.enemies[:]:
            enemy['pos'].x -= self.enemy_speed
            enemy['time'] += 1
            dy = enemy['type']['amp'] * math.sin(enemy['type']['freq'] * enemy['time'] + enemy['type']['offset'])
            enemy['pos'].y = enemy['base_y'] + dy
            
            # Clamp to screen to avoid weird loops
            enemy['pos'].y = np.clip(enemy['pos'].y, self.ENEMY_RADIUS, self.SCREEN_HEIGHT - self.ENEMY_RADIUS)

            if enemy['pos'].x < -self.ENEMY_RADIUS:
                self.enemies.remove(enemy)

    def _generate_stardust(self):
        stardust_list = []
        for _ in range(self.STARDUST_COUNT):
            pos = pygame.Vector2(
                self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
                self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
            )
            rect = pygame.Rect(pos.x - self.STARDUST_RADIUS, pos.y - self.STARDUST_RADIUS, self.STARDUST_RADIUS*2, self.STARDUST_RADIUS*2)
            stardust_list.append({'rect': rect, 'pos': pos, 'phase': self.np_random.uniform(0, 2 * math.pi)})
        return stardust_list
        
    def _create_stars(self, count):
        stars = []
        for _ in range(count):
            pos = (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT))
            radius = random.choice([1, 1, 1, 2])
            brightness = random.randint(50, 150)
            color = (brightness, brightness, int(brightness*1.2)) # Bluish tint
            stars.append({'pos': pos, 'radius': radius, 'color': color})
        return stars

    def _create_particles(self, pos, color, count, max_speed, life=40):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'color': color,
                'life': self.np_random.integers(life // 2, life),
                'max_life': life
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "shield_energy": self.shield_energy,
            "is_shielded": self.shield_active_timer > 0,
        }

    def _render_game(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        for star in self._background_stars:
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['radius'])

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['life']/2, p['life']/2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['life']/4, p['life']/4), p['life']/4)
            self.screen.blit(s, (int(p['pos'].x - p['life']/4), int(p['pos'].y - p['life']/4)), special_flags=pygame.BLEND_RGBA_ADD)

        # Stardust
        for dust in self.stardust:
            glow_size = self.STARDUST_RADIUS * (2 + 0.3 * math.sin(pygame.time.get_ticks() * 0.005 + dust['phase']))
            self._draw_glow(dust['pos'], self.COLOR_STARDUST_GLOW, glow_size)
            pygame.draw.circle(self.screen, self.COLOR_STARDUST, (int(dust['pos'].x), int(dust['pos'].y)), self.STARDUST_RADIUS)

        # Enemies
        for enemy in self.enemies:
            self._draw_glow(enemy['pos'], self.COLOR_ENEMY_GLOW, self.ENEMY_RADIUS * 2.5)
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, (int(enemy['pos'].x), int(enemy['pos'].y)), self.ENEMY_RADIUS)
            
        # Player
        if not self.game_over:
            # Shield effect
            if self.shield_active_timer > 0:
                shield_ratio = self.shield_active_timer / self.SHIELD_DURATION
                radius = self.PLAYER_RADIUS * 1.5 * (1 + 0.1 * math.sin(pygame.time.get_ticks() * 0.1))
                alpha = int(150 * shield_ratio)
                color = (*self.COLOR_SHIELD, alpha)
                self._draw_glow(self.player_pos, color, radius * 1.5)
                pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), int(radius), color)

            # Player ship
            self._draw_glow(self.player_pos, self.COLOR_PLAYER_GLOW, self.PLAYER_RADIUS * 3)
            p1 = (self.player_pos.x + self.PLAYER_RADIUS, self.player_pos.y)
            p2 = (self.player_pos.x - self.PLAYER_RADIUS, self.player_pos.y - self.PLAYER_RADIUS*0.8)
            p3 = (self.player_pos.x - self.PLAYER_RADIUS, self.player_pos.y + self.PLAYER_RADIUS*0.8)
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3])
            
    def _draw_glow(self, pos, color, radius):
        s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, color, (radius, radius), radius)
        self.screen.blit(s, (int(pos.x - radius), int(pos.y - radius)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Shield Energy Bar
        bar_width = 200
        bar_height = 15
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        
        energy_ratio = self.shield_energy / self.MAX_SHIELD_ENERGY
        
        pygame.draw.rect(self.screen, self.COLOR_UI_SHIELD_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_SHIELD_BAR, (bar_x, bar_y, int(bar_width * energy_ratio), bar_height))
        
        shield_text = self.font_ui.render("SHIELD", True, self.COLOR_UI_TEXT)
        self.screen.blit(shield_text, (bar_x - shield_text.get_width() - 10, 8))

        if self.game_over:
            over_text = self.font_game_over.render("GAME OVER", True, self.COLOR_ENEMY)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # For human play, we need a real display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Haunted Nebula")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # We need to track key presses for the "press" logic (not just held)
    prev_shift_pressed = False

    while not terminated:
        # --- Human Controls ---
        # Note: This is a simplified mapping for human play.
        # The agent uses the MultiDiscrete action space directly.
        current_keys = pygame.key.get_pressed()
        
        # Movement action (ignored by this env)
        movement_action = 0 
        
        # Space action (shield)
        space_action = 1 if current_keys[pygame.K_SPACE] else 0
        
        # Shift action (gravity flip)
        shift_action = 1 if current_keys[pygame.K_LSHIFT] or current_keys[pygame.K_RSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Run at 30 FPS
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()