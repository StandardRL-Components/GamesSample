import os
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame


# Set the SDL video driver to "dummy" for headless operation
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.ARENA_MARGIN = 20
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_ENEMY = (255, 50, 100)
        self.COLOR_ENEMY_GLOW = (255, 50, 100, 50)
        self.COLOR_P_PROJECTILE = (255, 255, 0)
        self.COLOR_E_PROJECTILE = (255, 150, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_ARENA = (50, 80, 120)

        # Player Physics
        self.PLAYER_ACCEL = 0.4
        self.PLAYER_BRAKE = 0.6
        self.PLAYER_TURN_RATE = 4.0
        self.PLAYER_MAX_SPEED = 5.0
        self.PLAYER_FRICTION = 0.96
        self.PLAYER_DRIFT_FRICTION = 0.98
        self.PLAYER_DRIFT_TURN_MULT = 1.5
        self.PLAYER_SHOOT_COOLDOWN = 8 # frames

        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 20)
            self.font_stage = pygame.font.Font(pygame.font.get_default_font(), 24)
        except IOError:
            self.font_ui = pygame.font.SysFont("monospace", 20)
            self.font_stage = pygame.font.SysFont("monospace", 24)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1

        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = 0.0
        self.player_health = 0
        self.player_shoot_timer = 0

        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

    def _get_random_pos_in_arena(self):
        return pygame.math.Vector2(
            self.np_random.uniform(self.ARENA_MARGIN + 20, self.WIDTH - self.ARENA_MARGIN - 20),
            self.np_random.uniform(self.ARENA_MARGIN + 20, self.HEIGHT - self.ARENA_MARGIN - 20)
        )

    def _spawn_enemies(self):
        self.enemies.clear()
        num_enemies = 5
        for _ in range(num_enemies):
            pos = self._get_random_pos_in_arena()
            while self.player_pos.distance_to(pos) < 150: # Avoid spawning on player
                pos = self._get_random_pos_in_arena()

            ai_type = self.np_random.choice(['circle', 'patrol'])
            ai_state = {}
            if ai_type == 'circle':
                ai_state['center'] = pos.copy()
                ai_state['radius'] = self.np_random.uniform(40, 80)
                ai_state['angle'] = self.np_random.uniform(0, 360)
                ai_state['speed'] = self.np_random.uniform(0.5, 1.5)
            elif ai_type == 'patrol':
                ai_state['pt_a'] = pos + pygame.math.Vector2(self.np_random.uniform(-100, 100), self.np_random.uniform(-100, 100))
                ai_state['pt_b'] = pos + pygame.math.Vector2(self.np_random.uniform(-100, 100), self.np_random.uniform(-100, 100))
                ai_state['target'] = 'b'
            
            enemy_fire_rate = max(10, 20 - (self.stage - 1) * 2)

            self.enemies.append({
                'pos': pos,
                'health': 10,
                'max_health': 10,
                'fire_timer': self.np_random.integers(0, enemy_fire_rate),
                'fire_rate': enemy_fire_rate,
                'ai_type': ai_type,
                'ai_state': ai_state,
                'hit_timer': 0,
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1

        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = -90
        self.player_health = 30
        self.player_shoot_timer = 0

        self.enemies.clear()
        self.player_projectiles.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()
        
        self._spawn_enemies()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.02  # Inaction penalty
        terminated = False
        truncated = False

        # --- Action Handling & Player Update ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        dist_before = float('inf')
        nearest_enemy = None
        if self.enemies:
            nearest_enemy = min(self.enemies, key=lambda e: self.player_pos.distance_squared_to(e['pos']))
            dist_before = self.player_pos.distance_to(nearest_enemy['pos'])

        # Turning
        turn_rate = self.PLAYER_TURN_RATE * (self.PLAYER_DRIFT_TURN_MULT if shift_held else 1.0)
        if movement == 3: # Left
            self.player_angle -= turn_rate
        if movement == 4: # Right
            self.player_angle += turn_rate
        
        # Acceleration
        accel_vec = pygame.math.Vector2(0, 0)
        if movement == 1: # Up
            accel_vec = pygame.math.Vector2(1, 0).rotate(self.player_angle) * self.PLAYER_ACCEL
        elif movement == 2: # Down
            if self.player_vel.length() > 0.1:
                accel_vec = -self.player_vel.normalize() * self.PLAYER_BRAKE
        
        self.player_vel += accel_vec
        
        # Friction
        friction = self.PLAYER_DRIFT_FRICTION if shift_held else self.PLAYER_FRICTION
        self.player_vel *= friction
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)

        # Update position
        self.player_pos += self.player_vel
        
        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, self.ARENA_MARGIN, self.WIDTH - self.ARENA_MARGIN)
        self.player_pos.y = np.clip(self.player_pos.y, self.ARENA_MARGIN, self.HEIGHT - self.ARENA_MARGIN)
        
        # Movement reward
        if self.enemies and nearest_enemy:
            dist_after = self.player_pos.distance_to(nearest_enemy['pos'])
            if dist_after < dist_before - 0.1:
                reward += 0.1
            elif dist_after > dist_before + 0.1:
                reward -= 0.1

        # Shooting
        if self.player_shoot_timer > 0:
            self.player_shoot_timer -= 1
        if space_held and self.player_shoot_timer == 0:
            self.player_shoot_timer = self.PLAYER_SHOOT_COOLDOWN
            proj_vel = pygame.math.Vector2(1, 0).rotate(self.player_angle) * 8.0
            self.player_projectiles.append({'pos': self.player_pos.copy(), 'vel': proj_vel})
            if nearest_enemy:
                vec_to_enemy = nearest_enemy['pos'] - self.player_pos
                if vec_to_enemy.length() > 0:
                    angle_diff = proj_vel.angle_to(vec_to_enemy)
                    if abs(angle_diff) < 15:
                        reward += 2.0

        # --- Update Game Objects ---
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        # --- Stage & Termination Check ---
        if not self.enemies:
            reward += 100
            if self.stage < 3:
                self.stage += 1
                self._spawn_enemies()
                self._create_explosion(self.player_pos, 20, self.COLOR_UI_TEXT, 1.5)
            else:
                terminated = True # Game won
                self.game_over = True

        if self.player_health <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            self._create_explosion(self.player_pos, 100, self.COLOR_PLAYER)

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_projectiles(self):
        reward = 0
        # Player projectiles
        for p in self.player_projectiles[:]:
            p['pos'] += p['vel']
            if not self.screen.get_rect().inflate(20, 20).collidepoint(p['pos']):
                self.player_projectiles.remove(p)
                continue
            
            for e in self.enemies[:]:
                if e['pos'].distance_to(p['pos']) < 12: # Hit radius
                    if p in self.player_projectiles:
                        self.player_projectiles.remove(p)
                    e['health'] -= 1
                    e['hit_timer'] = 5 # Flash on hit
                    self._create_explosion(p['pos'], 5, self.COLOR_P_PROJECTILE)
                    if e['health'] <= 0:
                        reward += 10
                        self._create_explosion(e['pos'], 50, self.COLOR_ENEMY)
                        if e in self.enemies:
                            self.enemies.remove(e)
                    break
        return reward

    def _update_enemies(self):
        reward = 0
        enemy_proj_speed = 3.0 + (self.stage - 1) * 0.2
        for e in self.enemies:
            # AI Movement
            if e['ai_type'] == 'circle':
                e['ai_state']['angle'] += e['ai_state']['speed']
                offset = pygame.math.Vector2(e['ai_state']['radius'], 0).rotate(e['ai_state']['angle'])
                e['pos'] = e['ai_state']['center'] + offset
            elif e['ai_type'] == 'patrol':
                target_pos = e['ai_state']['pt_a'] if e['ai_state']['target'] == 'a' else e['ai_state']['pt_b']
                if e['pos'].distance_to(target_pos) < 5:
                    e['ai_state']['target'] = 'a' if e['ai_state']['target'] == 'b' else 'b'
                direction = (target_pos - e['pos']).normalize() if (target_pos - e['pos']).length() > 0 else pygame.math.Vector2(0,0)
                e['pos'] += direction * 1.0
            
            e['pos'].x = np.clip(e['pos'].x, self.ARENA_MARGIN, self.WIDTH - self.ARENA_MARGIN)
            e['pos'].y = np.clip(e['pos'].y, self.ARENA_MARGIN, self.HEIGHT - self.ARENA_MARGIN)

            if e['hit_timer'] > 0:
                e['hit_timer'] -= 1

            # AI Shooting
            e['fire_timer'] -= 1
            if e['fire_timer'] <= 0:
                e['fire_timer'] = e['fire_rate']
                if self.player_pos.distance_to(e['pos']) < 300: # Range
                    direction = (self.player_pos - e['pos']).normalize()
                    self.enemy_projectiles.append({'pos': e['pos'].copy(), 'vel': direction * enemy_proj_speed})

        # Enemy projectiles
        for p in self.enemy_projectiles[:]:
            p['pos'] += p['vel']
            if not self.screen.get_rect().inflate(20, 20).collidepoint(p['pos']):
                self.enemy_projectiles.remove(p)
                continue
            
            if self.player_pos.distance_to(p['pos']) < 12:
                if p in self.enemy_projectiles:
                    self.enemy_projectiles.remove(p)
                self.player_health -= 1
                reward -= 1
                self._create_explosion(p['pos'], 10, self.COLOR_E_PROJECTILE)
                break
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['radius'] -= p['decay']
            if p['radius'] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 4) * speed_mult
            vel = pygame.math.Vector2(speed, 0).rotate(angle)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': self.np_random.uniform(2, 6),
                'decay': self.np_random.uniform(0.05, 0.2),
                'color': color
            })

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
            "health": self.player_health,
            "stage": self.stage,
        }

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (self.ARENA_MARGIN, self.ARENA_MARGIN, self.WIDTH - 2*self.ARENA_MARGIN, self.HEIGHT - 2*self.ARENA_MARGIN), 2)
        
        self._render_particles()
        self._render_projectiles()
        self._render_enemies()
        self._render_player()

    def _render_particles(self):
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])

    def _render_projectiles(self):
        for p in self.player_projectiles:
            pos1 = p['pos']
            pos2 = p['pos'] - p['vel'].normalize() * 8
            pygame.draw.line(self.screen, self.COLOR_P_PROJECTILE, (int(pos1.x), int(pos1.y)), (int(pos2.x), int(pos2.y)), 3)
        for p in self.enemy_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), 4, self.COLOR_E_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), 4, self.COLOR_E_PROJECTILE)

    def _render_enemies(self):
        for e in self.enemies:
            size = 10
            rect = pygame.Rect(e['pos'].x - size, e['pos'].y - size, size*2, size*2)
            color = (255, 255, 255) if e['hit_timer'] > 0 else self.COLOR_ENEMY
            pygame.gfxdraw.filled_circle(self.screen, int(e['pos'].x), int(e['pos'].y), 18, self.COLOR_ENEMY_GLOW)
            pygame.draw.rect(self.screen, color, rect)
            health_pct = e['health'] / e['max_health']
            bar_width = 20
            pygame.draw.rect(self.screen, (255,0,0), (e['pos'].x - bar_width/2, e['pos'].y - 20, bar_width, 4))
            pygame.draw.rect(self.screen, (0,255,0), (e['pos'].x - bar_width/2, e['pos'].y - 20, bar_width * health_pct, 4))

    def _render_player(self):
        if self.player_health <= 0: return

        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), 25, self.COLOR_PLAYER_GLOW)
        
        size = 12
        p1 = self.player_pos + pygame.math.Vector2(size, 0).rotate(self.player_angle)
        p2 = self.player_pos + pygame.math.Vector2(-size/2, size * 0.8).rotate(self.player_angle)
        p3 = self.player_pos + pygame.math.Vector2(-size/2, -size * 0.8).rotate(self.player_angle)
        points = [(int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (int(p3.x), int(p3.y))]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        health_text = self.font_ui.render(f"HEALTH: {max(0, self.player_health)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 10))

        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        stage_text = self.font_stage.render(f"STAGE {self.stage}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (self.WIDTH/2 - stage_text.get_width()/2, 10))

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly.
    # To do so, you might need to comment out the line:
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    # at the top of the file.
    
    # Re-enable display for human play
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
    
    pygame.display.init()
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robot Arena")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        if truncated:
            terminated = True # End loop if truncated
        
        # --- Render to Screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

    env.close()