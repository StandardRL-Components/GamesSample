import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:02:20.958036
# Source Brief: brief_01484.md
# Brief Index: 1484
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A minimalist tower defense game where the player uses time portals
    to defend a central tower from waves of enemies.

    The player controls a cursor to place an entry and exit portal. Enemies
    that touch the entry portal are teleported to the exit. After several
    teleportations, an enemy is destabilized and destroyed. The player also
    has a global time-slowing ability with a cooldown.

    The goal is to survive as many waves as possible.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    game_description = (
        "Defend your central tower from waves of enemies by placing time portals. "
        "Enemies that pass through portals enough times are destroyed."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place a portal and shift to activate a time-slowing warp."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    TOTAL_WAVES = 20

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 45, 60)
    COLOR_TOWER = (50, 200, 100)
    COLOR_TOWER_DAMAGE = (255, 100, 100)
    COLOR_ENEMY = (255, 60, 60)
    COLOR_PORTAL_1 = (0, 150, 255)
    COLOR_PORTAL_2 = (255, 150, 0)
    COLOR_PORTAL_LINK = (150, 150, 150, 100)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TIME_SLOW_AURA = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_COOLDOWN_BAR_BG = (50, 50, 70)
    COLOR_COOLDOWN_BAR_FG = (100, 180, 255)

    # Game Parameters
    TOWER_POS = pygame.Vector2(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    TOWER_RADIUS = 25
    TOWER_MAX_HEALTH = 100
    
    ENEMY_RADIUS = 8
    ENEMY_BASE_SPEED = 0.8
    ENEMY_DAMAGE = 10
    ENEMY_PASSES_TO_DESTROY = 3

    PORTAL_RADIUS = 20
    CURSOR_SPEED = 10
    
    TIME_SLOW_DURATION = 90  # 3 seconds at 30 FPS
    TIME_SLOW_COOLDOWN = 300 # 10 seconds at 30 FPS
    TIME_SLOW_FACTOR = 0.4

    WAVE_INTERMISSION_TIME = 90 # 3 seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        if self.render_mode == "human":
            self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # State variables are initialized in reset()
        self.tower_health = 0
        self.score = 0
        self.steps = 0
        self.wave = 0
        self.enemies = []
        self.particles = []
        self.portals = [None, None]
        self.portal_placement_index = 0
        self.portal_cursor_pos = pygame.Vector2(0, 0)
        self.time_slow_timer = 0
        self.time_slow_cooldown_timer = 0
        self.wave_intermission_timer = 0
        self.prev_space_held = False
        self.prev_shift_held = False

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.tower_health = self.TOWER_MAX_HEALTH
        
        self.enemies.clear()
        self.particles.clear()
        
        self.portals = [None, None]
        self.portal_placement_index = 0
        self.portal_cursor_pos = pygame.Vector2(self.SCREEN_WIDTH * 0.25, self.SCREEN_HEIGHT * 0.5)

        self.time_slow_timer = 0
        self.time_slow_cooldown_timer = 0
        
        self.wave = 0
        self.wave_intermission_timer = self.WAVE_INTERMISSION_TIME // 2

        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        reward = 0.01 # Small reward for surviving
        
        self._handle_input(movement, space_press, shift_press)
        
        self._update_timers()
        self._update_particles()
        
        damage_dealt, enemies_destroyed = self._update_enemies()
        if damage_dealt > 0:
            self.tower_health -= damage_dealt
            reward -= 5.0 * (damage_dealt / self.ENEMY_DAMAGE) # Penalize for damage
        if enemies_destroyed > 0:
            self.score += enemies_destroyed
            reward += 1.0 * enemies_destroyed # Reward for destroying enemies

        self._check_wave_state()
        
        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated:
            if self.tower_health <= 0:
                reward -= 100.0 # Large penalty for losing
            elif self.wave > self.TOTAL_WAVES:
                reward += 100.0 # Large reward for winning

        if self.render_mode == "human":
            self._render_frame()

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_press, shift_press):
        # Move cursor
        if movement == 1: self.portal_cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.portal_cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.portal_cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.portal_cursor_pos.x += self.CURSOR_SPEED
        self.portal_cursor_pos.x = np.clip(self.portal_cursor_pos.x, 0, self.SCREEN_WIDTH)
        self.portal_cursor_pos.y = np.clip(self.portal_cursor_pos.y, 0, self.SCREEN_HEIGHT)

        # Place portals
        if space_press:
            # SFX: Portal Placement Sound
            self.portals[self.portal_placement_index] = self.portal_cursor_pos.copy()
            self.portal_placement_index = (self.portal_placement_index + 1) % 2
            
        # Activate time slow
        if shift_press and self.time_slow_cooldown_timer <= 0:
            # SFX: Time Slow Activation
            self.time_slow_timer = self.TIME_SLOW_DURATION
            self.time_slow_cooldown_timer = self.TIME_SLOW_COOLDOWN

    def _update_timers(self):
        if self.time_slow_timer > 0:
            self.time_slow_timer -= 1
        if self.time_slow_cooldown_timer > 0:
            self.time_slow_cooldown_timer -= 1
        if self.wave_intermission_timer > 0:
            self.wave_intermission_timer -= 1

    def _update_enemies(self):
        damage_dealt = 0
        enemies_destroyed = 0
        
        for enemy in reversed(self.enemies):
            # Movement
            direction = (self.TOWER_POS - enemy['pos']).normalize()
            speed = self.ENEMY_BASE_SPEED + 0.05 * math.floor(self.wave / 5)
            if self.time_slow_timer > 0:
                speed *= self.TIME_SLOW_FACTOR
            enemy['pos'] += direction * speed

            # Update cooldowns
            if enemy['teleport_cooldown'] > 0:
                enemy['teleport_cooldown'] -= 1

            # Check portal collision
            p1, p2 = self.portals
            if p1 and p2 and enemy['teleport_cooldown'] == 0:
                if enemy['pos'].distance_to(p1) < self.PORTAL_RADIUS:
                    # SFX: Teleport Whoosh
                    enemy['pos'] = p2.copy()
                    enemy['teleport_cooldown'] = 30 # 1s cooldown
                    enemy['passes'] += 1
                    if enemy['passes'] >= self.ENEMY_PASSES_TO_DESTROY:
                        self._create_explosion(enemy['pos'], self.COLOR_ENEMY)
                        self.enemies.remove(enemy)
                        enemies_destroyed += 1
                        continue

            # Check tower collision
            if enemy['pos'].distance_to(self.TOWER_POS) < self.TOWER_RADIUS + self.ENEMY_RADIUS:
                # SFX: Tower Hit Impact
                damage_dealt += self.ENEMY_DAMAGE
                self._create_explosion(enemy['pos'], self.COLOR_TOWER_DAMAGE)
                self.enemies.remove(enemy)
                continue
        
        return damage_dealt, enemies_destroyed

    def _check_wave_state(self):
        if not self.enemies and self.wave <= self.TOTAL_WAVES:
            if self.wave_intermission_timer <= 0:
                self._start_new_wave()
                self.wave_intermission_timer = self.WAVE_INTERMISSION_TIME

    def _start_new_wave(self):
        self.wave += 1
        if self.wave > self.TOTAL_WAVES:
            return
            
        # SFX: New Wave Horn
        num_enemies = 3 + math.floor(self.wave / 2)
        
        for _ in range(num_enemies):
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ENEMY_RADIUS)
            elif edge == 1: # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ENEMY_RADIUS)
            elif edge == 2: # Left
                pos = pygame.Vector2(-self.ENEMY_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            else: # Right
                pos = pygame.Vector2(self.SCREEN_WIDTH + self.ENEMY_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            
            self.enemies.append({'pos': pos, 'teleport_cooldown': 0, 'passes': 0})
            
    def _create_explosion(self, pos, color):
        # SFX: Explosion
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return (
            self.tower_health <= 0 or
            self.wave > self.TOTAL_WAVES
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "tower_health": self.tower_health,
        }

    def _render_frame(self):
        if self.render_mode == "human":
            self.window.blit(self.screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def _render_game(self):
        self._draw_background_grid()
        self._draw_portals()
        self._draw_tower()
        self._draw_enemies()
        self._draw_particles()
        self._draw_cursor()

    def _draw_background_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_glowing_circle(self, surface, pos, radius, color, max_glow=10):
        pos_int = (int(pos.x), int(pos.y))
        for i in range(max_glow, 0, -1):
            alpha = int(100 * (1 - (i / max_glow)))
            glow_color = (*color, alpha)
            pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], int(radius + i), glow_color)

    def _draw_portals(self):
        p1, p2 = self.portals
        if p1 and p2:
            pygame.draw.aaline(self.screen, self.COLOR_PORTAL_LINK, p1, p2, 1)
        if p1:
            self._draw_glowing_circle(self.screen, p1, self.PORTAL_RADIUS, self.COLOR_PORTAL_1, 15)
        if p2:
            self._draw_glowing_circle(self.screen, p2, self.PORTAL_RADIUS, self.COLOR_PORTAL_2, 15)
    
    def _draw_tower(self):
        pos_int = (int(self.TOWER_POS.x), int(self.TOWER_POS.y))
        self._draw_glowing_circle(self.screen, self.TOWER_POS, self.TOWER_RADIUS, self.COLOR_TOWER, 20)
        
        # Health bar
        health_percent = self.tower_health / self.TOWER_MAX_HEALTH
        bar_width = 60
        bar_height = 8
        bar_x = self.TOWER_POS.x - bar_width / 2
        bar_y = self.TOWER_POS.y - self.TOWER_RADIUS - 20
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height))
        fill_color = self.COLOR_TOWER if health_percent > 0.3 else self.COLOR_TOWER_DAMAGE
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, bar_width * health_percent, bar_height))

    def _draw_enemies(self):
        for enemy in self.enemies:
            pos_int = (int(enemy['pos'].x), int(enemy['pos'].y))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)
            
            if self.time_slow_timer > 0:
                aura_radius = self.ENEMY_RADIUS + 4 + 2 * math.sin(self.steps * 0.2)
                self._draw_glowing_circle(self.screen, enemy['pos'], aura_radius, self.COLOR_TIME_SLOW_AURA, 5)

    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['lifetime'] / 30.0))
            color_with_alpha = (*p['color'], alpha)
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(p['radius']), color_with_alpha)

    def _draw_cursor(self):
        pos = self.portal_cursor_pos
        size = 10
        pygame.draw.aaline(self.screen, self.COLOR_CURSOR, (pos.x - size, pos.y), (pos.x + size, pos.y))
        pygame.draw.aaline(self.screen, self.COLOR_CURSOR, (pos.x, pos.y - size), (pos.x, pos.y + size))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Wave
        wave_str = f"WAVE: {self.wave}/{self.TOTAL_WAVES}"
        if not self.enemies and self.wave <= self.TOTAL_WAVES:
            wave_str = f"WAVE {self.wave+1} IN {math.ceil(self.wave_intermission_timer / self.metadata['render_fps'])}"
        elif self.wave > self.TOTAL_WAVES:
            wave_str = "ALL WAVES CLEARED!"

        wave_text = self.font_large.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Time Slow Cooldown
        cooldown_text = self.font_small.render("TIME WARP [SHIFT]", True, self.COLOR_UI_TEXT)
        self.screen.blit(cooldown_text, (10, self.SCREEN_HEIGHT - 30))
        
        bar_width = 150
        bar_height = 10
        bar_x = 10
        bar_y = self.SCREEN_HEIGHT - 15

        pygame.draw.rect(self.screen, self.COLOR_COOLDOWN_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        
        cooldown_percent = 1.0
        if self.TIME_SLOW_COOLDOWN > 0:
            cooldown_percent = 1.0 - (self.time_slow_cooldown_timer / self.TIME_SLOW_COOLDOWN)

        if self.time_slow_cooldown_timer > 0:
            fill_width = bar_width * cooldown_percent
            pygame.draw.rect(self.screen, self.COLOR_COOLDOWN_BAR_FG, (bar_x, bar_y, fill_width, bar_height), border_radius=3)
        else:
            pygame.draw.rect(self.screen, self.COLOR_TIME_SLOW_AURA, (bar_x, bar_y, bar_width, bar_height), border_radius=3)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # Example usage and human play
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # --- Human Controls ---
    # Arrows: Move cursor
    # Space: Place portal
    # Shift: Activate time warp
    # R: Reset environment
    
    while True:
        action = [0, 0, 0] # [movement, space, shift]
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        if keys[pygame.K_r]:
            print("Resetting environment.")
            obs, info = env.reset()
            terminated = False
            truncated = False

        if not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                quit()

        if terminated or truncated:
            font = pygame.font.SysFont("Consolas", 48, bold=True)
            win_status = "YOU WON!" if info['tower_health'] > 0 and not truncated else "GAME OVER"
            text = font.render(win_status, True, (255, 255, 255))
            text_rect = text.get_rect(center=(env.SCREEN_WIDTH/2, env.SCREEN_HEIGHT/2))
            env.screen.blit(text, text_rect)
            
            font_small = pygame.font.SysFont("Consolas", 24, bold=True)
            reset_text = font_small.render("Press 'R' to Restart", True, (200, 200, 200))
            reset_rect = reset_text.get_rect(center=(env.SCREEN_WIDTH/2, env.SCREEN_HEIGHT/2 + 40))
            env.screen.blit(reset_text, reset_rect)

            env._render_frame()