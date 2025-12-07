import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:23:23.213889
# Source Brief: brief_00440.md
# Brief Index: 440
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
        "A steampunk-themed side-scroller where you control a sprocket-like character. "
        "Master the art of flipping gravity to navigate treacherous platforms and race to the finish line."
    )
    user_guide = (
        "Controls: ←→ to move left and right. Press space to flip gravity. "
        "Press shift to use a collected power-up."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000

    # Colors (Steampunk Theme)
    COLOR_BG = (20, 25, 30)
    COLOR_BG_GEAR = (30, 35, 40)
    COLOR_TRACK = (100, 80, 60)
    COLOR_TRACK_HIGHLIGHT = (130, 110, 90)
    
    COLOR_PLAYER = (0, 191, 255) # Bright Blue
    COLOR_PLAYER_GLOW = (0, 191, 255, 50)
    
    COLOR_ENEMY = (255, 50, 50) # Bright Red
    COLOR_ENEMY_EYE = (255, 200, 200)
    
    COLOR_OBSTACLE = (200, 40, 40)
    
    COLOR_POWERUP = (50, 255, 150) # Bright Green
    COLOR_POWERUP_GLOW = (50, 255, 150, 60)
    
    COLOR_FINISH = (255, 215, 0) # Gold
    
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    
    # Physics
    GRAVITY = 0.8
    PLAYER_ACCEL = 1.2
    PLAYER_FRICTION = -0.08
    PLAYER_MAX_SPEED = 12
    PLAYER_JUMP_STRENGTH = 12 # Not used but good to have
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 36)
        
        self.render_mode = render_mode
        self.best_time = float('inf')
        
        # self.reset() is called by the wrapper/runner
        # self.validate_implementation() # Not needed in production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.terminated = False
        
        self.player = self._Player(self.WIDTH / 4, self.HEIGHT / 2)
        
        self.gravity_direction = 1
        self.gravity_flip_cooldown = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.equipped_powerup = None
        self.powerup_timer = 0

        self.camera_x = 0
        
        self._generate_track()
        self.particles = []
        self.bg_gears = self._generate_bg_gears()
        
        self.start_time = pygame.time.get_ticks()
        self.current_time = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Action Handling ---
        # Movement
        if movement == 3: # Left
            self.player.vel.x += -self.PLAYER_ACCEL
        elif movement == 4: # Right
            self.player.vel.x += self.PLAYER_ACCEL
            
        # Gravity Flip (on press)
        if space_held and not self.prev_space_held and self.gravity_flip_cooldown <= 0:
            self.gravity_direction *= -1
            self.gravity_flip_cooldown = 30 # 1 second cooldown
            # sfx: GravityFlip.wav
            self._create_particles(self.player.pos, self.COLOR_PLAYER, 30, 8)
        
        # Power-up (on press)
        if shift_held and not self.prev_shift_held and self.equipped_powerup:
            if self.equipped_powerup == 'speed_boost':
                self.powerup_timer = 150 # 5 seconds
                # sfx: SpeedBoostActivate.wav
            self.equipped_powerup = None

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Game Logic Update ---
        prev_vel_x = self.player.vel.x
        self._update_game_state()
        
        # --- Reward Calculation ---
        reward += self.player.vel.x * 0.01 # Reward for forward velocity
        if self.player.vel.x < prev_vel_x and movement != 3: # Penalty for slowing down
            reward -= 0.1

        # --- Collision & Event Checks ---
        # Obstacles/Enemies
        for entity in self.enemies + self.obstacles:
            if self.player.rect.colliderect(entity['rect']):
                reward = -10
                self.terminated = True
                # sfx: PlayerHit.wav
                self._create_particles(self.player.pos, self.COLOR_ENEMY, 50, 10)
                break
        
        # Power-ups
        for powerup in self.powerups[:]:
            if self.player.rect.colliderect(powerup['rect']):
                self.equipped_powerup = powerup['type']
                self.powerups.remove(powerup)
                reward += 1
                # sfx: PowerupCollect.wav
                self._create_particles(powerup['rect'].center, self.COLOR_POWERUP, 20, 5)
                break
                
        # Finish Line
        if self.player.rect.colliderect(self.finish_line):
            self.current_time = (pygame.time.get_ticks() - self.start_time) / 1000
            if self.current_time < self.best_time:
                reward += 100 # New best time
                self.best_time = self.current_time
            else:
                reward += 50 # Completed track
            self.terminated = True
            # sfx: Win.wav

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.terminated = True
            reward -= 5 # Penalty for timeout
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self):
        # Update timers
        self.gravity_flip_cooldown = max(0, self.gravity_flip_cooldown - 1)
        self.powerup_timer = max(0, self.powerup_timer - 1)
        if not self.terminated:
            self.current_time = (pygame.time.get_ticks() - self.start_time) / 1000

        # Update Player
        effective_max_speed = self.PLAYER_MAX_SPEED * 1.5 if self.powerup_timer > 0 else self.PLAYER_MAX_SPEED
        self.player.update(self.gravity_direction * self.GRAVITY, self.platforms, effective_max_speed)
        self.player.pos.y = np.clip(self.player.pos.y, -50, self.HEIGHT + 50)
        
        # Update Enemies
        for enemy in self.enemies:
            enemy['pos'][0] += enemy['vel']
            if enemy['pos'][0] < enemy['path'][0] or enemy['pos'][0] > enemy['path'][1]:
                enemy['vel'] *= -1
            enemy['rect'].centerx = int(enemy['pos'][0])

        # Update Particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Update Camera
        target_camera_x = self.player.pos.x - self.WIDTH / 3
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for gear in self.bg_gears:
            self._draw_gear(self.screen, 
                           (int(gear['x'] - self.camera_x * gear['parallax']), gear['y']),
                           gear['radius'], gear['teeth'], gear['color'], 
                           (self.steps + gear['offset']) * gear['speed'])

    def _render_game(self):
        # Draw trails if moving fast
        if abs(self.player.vel.x) > 5 or self.powerup_timer > 0:
            self.player.trail.append((self.player.pos.copy(), 30))
        
        for pos, life in self.player.trail[:]:
            alpha = int(255 * (life / 30))
            # glow_color = (*self.COLOR_PLAYER, alpha // 4)
            self._draw_sprocket(self.screen, pos - pygame.Vector2(self.camera_x, 0), self.player.size, self.player.angle, self.COLOR_PLAYER, alpha)
            self.player.trail[self.player.trail.index((pos, life))] = (pos, life - 1)
            if life <= 1:
                self.player.trail.pop(0)

        # Draw entities
        for item_list in [self.platforms, self.obstacles]:
            for item in item_list:
                self._draw_camera_adjusted(item['rect'], self.COLOR_TRACK if 'color' not in item else item['color'])
        
        for powerup in self.powerups:
            self._draw_glowing_circle(powerup['rect'].center, self.COLOR_POWERUP, self.COLOR_POWERUP_GLOW, 10)

        for enemy in self.enemies:
            r = enemy['rect'].copy()
            r.x -= int(self.camera_x)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, r, border_radius=3)
            eye_pos = (r.centerx + 5 * np.sign(enemy['vel']), r.centery)
            pygame.draw.circle(self.screen, self.COLOR_ENEMY_EYE, eye_pos, 3)

        pygame.draw.rect(self.screen, self.COLOR_FINISH, self.finish_line.move(-self.camera_x, 0))

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'].x - self.camera_x), int(p['pos'].y))
            alpha = max(0, int(255 * (p['life'] / p['max_life'])))
            p['color'].a = alpha
            pygame.draw.circle(self.screen, p['color'], pos, int(p['size'] * (p['life']/p['max_life'])))

        # Draw player on top
        player_screen_pos = self.player.pos - pygame.Vector2(self.camera_x, 0)
        for i in range(5, 0, -1):
             pygame.gfxdraw.aacircle(self.screen, int(player_screen_pos.x), int(player_screen_pos.y), self.player.size + i*2, self.COLOR_PLAYER_GLOW)
        self._draw_sprocket(self.screen, player_screen_pos, self.player.size, self.player.angle, self.COLOR_PLAYER)

    def _draw_sprocket(self, surface, pos, radius, angle, color, alpha=255):
        if alpha < 255:
            temp_surface = pygame.Surface((radius*2+2, radius*2+2), pygame.SRCALPHA)
            points = []
            num_teeth = 8
            for i in range(num_teeth * 2):
                r = radius if i % 2 == 0 else radius - 4
                a = angle + (i / (num_teeth * 2)) * 2 * math.pi
                points.append((radius+1 + r * math.cos(a), radius+1 + r * math.sin(a)))
            
            pygame.draw.polygon(temp_surface, (*color, alpha), points)
            pygame.draw.circle(temp_surface, (*color, alpha), (radius+1, radius+1), radius - 6)
            surface.blit(temp_surface, (int(pos.x - radius-1), int(pos.y - radius-1)))
        else:
            points = []
            num_teeth = 8
            for i in range(num_teeth * 2):
                r = radius if i % 2 == 0 else radius - 4
                a = angle + (i / (num_teeth * 2)) * 2 * math.pi
                points.append((pos.x + r * math.cos(a), pos.y + r * math.sin(a)))
            
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aacircle(surface, int(pos.x), int(pos.y), radius - 6, color)
            pygame.gfxdraw.filled_circle(surface, int(pos.x), int(pos.y), radius - 6, color)

    def _draw_gear(self, surface, pos, radius, teeth, color, angle):
        points = []
        for i in range(teeth * 2):
            r = radius if i % 2 == 0 else radius * 0.8
            a = angle + (i / (teeth * 2)) * 2 * math.pi
            points.append((pos[0] + r * math.cos(a), pos[1] + r * math.sin(a)))
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius * 0.6), self.COLOR_BG)

    def _render_ui(self):
        # Time
        time_text = f"TIME: {self.current_time:.2f}"
        self._draw_text(time_text, (10, 10), self.font_small)
        
        # Best Time
        best_time_str = f"BEST: {self.best_time:.2f}" if self.best_time != float('inf') else "BEST: N/A"
        self._draw_text(best_time_str, (10, 30), self.font_small)

        # Speed
        speed_val = abs(self.player.vel.x)
        speed_text = f"SPEED: {speed_val:.1f}"
        self._draw_text(speed_text, (self.WIDTH - 150, 10), self.font_small)

        # Power-up status
        if self.equipped_powerup:
            self._draw_text("P-UP:", (self.WIDTH - 150, 30), self.font_small)
            if self.equipped_powerup == 'speed_boost':
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, (self.WIDTH - 90, 35, 10, 10))
        if self.powerup_timer > 0:
            bar_width = 100 * (self.powerup_timer / 150)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (self.WIDTH - 150, 50, bar_width, 5))
            
        if self.terminated:
            if self.player.rect.colliderect(self.finish_line):
                msg = "TRACK COMPLETE"
            else:
                msg = "SYSTEM FAILURE"
            self._draw_text(msg, (self.WIDTH/2, self.HEIGHT/2 - 20), self.font_large, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, center=False):
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, color)
        
        if center:
            text_rect = text_surf.get_rect(center=pos)
        else:
            text_rect = text_surf.get_rect(topleft=pos)
        
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _draw_camera_adjusted(self, rect, color):
        r = rect.copy()
        r.x -= int(self.camera_x)
        pygame.draw.rect(self.screen, color, r)

    def _draw_glowing_circle(self, pos, color, glow_color, radius):
        screen_pos = (int(pos[0] - self.camera_x), int(pos[1]))
        for i in range(4, 0, -1):
            pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], radius + i * 2, glow_color)
        pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], radius, color)
        pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time": self.current_time,
            "best_time": self.best_time if self.best_time != float('inf') else -1,
            "player_pos": tuple(self.player.pos),
            "player_vel": tuple(self.player.vel),
        }

    def _generate_track(self):
        self.platforms = []
        self.enemies = []
        self.obstacles = []
        self.powerups = []
        
        x = 0
        y = self.HEIGHT * 3/4
        length = 200
        
        # Starting platform
        self.platforms.append({'rect': pygame.Rect(x-200, y, length+200, 20)})
        
        track_len = 10000
        while x < track_len:
            length = self.np_random.integers(200, 500)
            gap = self.np_random.integers(80, 150)
            y_change = self.np_random.integers(-100, 101)
            
            # Create platform
            new_y = np.clip(y + y_change, 100, self.HEIGHT - 100)
            platform_rect = pygame.Rect(x, y, length, 20)
            self.platforms.append({'rect': platform_rect})
            
            # Add enemies
            if self.np_random.random() < 0.4:
                path_start = x + 20
                path_end = x + length - 20
                enemy_y = y - 15
                self.enemies.append({
                    'rect': pygame.Rect(path_start, enemy_y - 10, 20, 20),
                    'pos': [path_start, enemy_y],
                    'vel': self.np_random.choice([-2, 2]),
                    'path': (path_start, path_end)
                })

            # Add obstacles (spikes)
            if self.np_random.random() < 0.3:
                for i in range(self.np_random.integers(1, 4)):
                    ox = x + self.np_random.integers(10, length - 10)
                    # Spikes on floor or ceiling
                    if self.np_random.random() < 0.5:
                        oy = y - 10 # on platform
                    else:
                        oy = y - 150 # on 'ceiling' above
                    self.obstacles.append({'rect': pygame.Rect(ox, oy, 10, 10), 'color': self.COLOR_OBSTACLE})

            # Add powerups
            if self.np_random.random() < 0.2:
                px = x + length / 2
                py = y - self.np_random.integers(40, 80)
                self.powerups.append({
                    'rect': pygame.Rect(px - 10, py - 10, 20, 20),
                    'type': 'speed_boost'
                })
            
            x += length + gap
            y = new_y
            
        self.finish_line = pygame.Rect(x, 0, 20, self.HEIGHT)
        self.track_width = x + 20

    def _generate_bg_gears(self):
        gears = []
        for _ in range(20):
            gears.append({
                'x': self.np_random.uniform(0, 15000),
                'y': self.np_random.uniform(0, self.HEIGHT),
                'radius': self.np_random.uniform(20, 100),
                'teeth': self.np_random.integers(6, 12),
                'color': (
                    self.np_random.integers(30, 50),
                    self.np_random.integers(35, 55),
                    self.np_random.integers(40, 60)
                ),
                'speed': self.np_random.uniform(-0.02, 0.02),
                'offset': self.np_random.uniform(0, 360),
                'parallax': self.np_random.uniform(0.1, 0.5)
            })
        return gears
        
    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': vel,
                'life': life,
                'max_life': life,
                'size': self.np_random.uniform(2, 5),
                'color': pygame.Color(*color)
            })

    def close(self):
        pygame.quit()

    class _Player:
        def __init__(self, x, y):
            self.size = 12
            self.pos = pygame.math.Vector2(x, y)
            self.vel = pygame.math.Vector2(0, 0)
            self.rect = pygame.Rect(x - self.size, y - self.size, self.size * 2, self.size * 2)
            self.on_ground = False
            self.angle = 0
            self.trail = []

        def update(self, gravity, platforms, max_speed):
            # Apply friction
            self.vel.x += self.vel.x * GameEnv.PLAYER_FRICTION
            if abs(self.vel.x) < 0.1: self.vel.x = 0
            self.vel.x = np.clip(self.vel.x, -max_speed, max_speed)

            # Apply gravity
            self.vel.y += gravity
            self.vel.y = np.clip(self.vel.y, -20, 20)

            # Move and collide
            self.on_ground = False
            self.pos.x += self.vel.x
            self.rect.centerx = int(self.pos.x)
            # No horizontal collision, can fall off

            self.pos.y += self.vel.y
            self.rect.centery = int(self.pos.y)
            for plat in platforms:
                if self.rect.colliderect(plat['rect']):
                    if gravity > 0: # Normal gravity
                        if self.vel.y > 0 and self.rect.bottom > plat['rect'].top and self.rect.top < plat['rect'].top:
                            self.rect.bottom = plat['rect'].top
                            self.pos.y = self.rect.centery
                            self.vel.y = 0
                            self.on_ground = True
                    else: # Flipped gravity
                        if self.vel.y < 0 and self.rect.top < plat['rect'].bottom and self.rect.bottom > plat['rect'].bottom:
                            self.rect.top = plat['rect'].bottom
                            self.pos.y = self.rect.centery
                            self.vel.y = 0
                            self.on_ground = True
            
            # Update rotation
            self.angle += self.vel.x * 0.05


if __name__ == '__main__':
    # This block is for human play testing and is not part of the official environment
    # It will not be executed by the test suite, but is useful for development.
    # To use, run `python your_file_name.py`
    
    # Allow rendering to the screen
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Gravity Sprocket")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # Create a mapping from keyboard keys to actions
    key_action_map = {
        pygame.K_LEFT: 3,
        pygame.K_a: 3,
        pygame.K_RIGHT: 4,
        pygame.K_d: 4,
    }
    
    while True:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        # --- Action Polling ---
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        
        for key, move_action in key_action_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize one direction if both are pressed
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

        # --- Game Over and Reset ---
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Time: {info['time']:.2f}s")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0