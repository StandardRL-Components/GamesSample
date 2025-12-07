import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A rhythm-based stealth game. Match beats to disable security systems, "
        "avoid detection by cameras, and infiltrate deeper into the facility."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to select a security tile. "
        "Press space in time with the rhythm bar to interact with the selected tile."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FINAL_LEVEL = 10
    MAX_STEPS = 5000
    FPS = 30

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_BG_ACCENT = (20, 40, 80)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_TILE = (50, 100, 200)
    COLOR_TILE_KEY = (200, 150, 50)
    COLOR_TILE_MATCHED = (70, 80, 100)
    COLOR_TILE_SELECTED = (255, 255, 255)
    COLOR_TILE_EXIT = (150, 50, 255)
    COLOR_CAMERA_CONE = (255, 200, 0)
    COLOR_CAMERA_DISABLED = (100, 100, 100)
    COLOR_CAMERA_BODY = (200, 200, 220)
    COLOR_DANGER = (255, 0, 0)
    COLOR_PULSE = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_RHYTHM_BAR = (100, 120, 150)
    COLOR_RHYTHM_HIT = (0, 255, 150)

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
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_huge = pygame.font.SysFont("Consolas", 64, bold=True)
        
        self.level = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.player_cover_pos = (0,0)
        self.player_vulnerable_timer = 0
        self.cameras = []
        self.tiles = []
        self.pulses = []
        self.particles = []
        self.selected_tile_idx = 0
        self.exit_active = False
        self.rhythm_timer = 0.0
        self.last_space_press = False
        self.bg_particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.level = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.last_space_press = False
        self.rhythm_timer = 0.0
        
        self._generate_bg_particles()
        self._generate_next_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, self.win, self.steps >= self.MAX_STEPS, self._get_info()

        movement, space_held, _ = action
        space_press = space_held and not self.last_space_press
        self.last_space_press = bool(space_held)
        
        self.steps += 1
        step_reward = 0

        self._update_timers()
        self._update_bg_particles()
        
        self._handle_input(movement, space_press)
        step_reward += self._update_game_state()

        terminated = self._is_terminated()
        truncated = self.steps >= self.MAX_STEPS

        if terminated:
            self.game_over = True
            if self.win:
                step_reward += 100.0
            else:
                step_reward += -100.0
        
        if truncated:
            self.game_over = True
        
        self.score += step_reward

        return (
            self._get_observation(),
            step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_next_level(self):
        self.level += 1
        self.tiles.clear()
        self.cameras.clear()
        self.pulses.clear()
        self.particles.clear()
        
        self.player_cover_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.player_vulnerable_timer = 0
        self.exit_active = False
        self.selected_tile_idx = 0

        num_tiles = 4 + (self.level - 1) // 2
        num_key_tiles = min(num_tiles, 1 + self.level // 2)
        cam_speed = 0.01 + self.level * 0.005
        num_cameras = 1 + self.level // 3

        if num_tiles > 0:
            angle_step = 360 / num_tiles
            radius = 100
            key_indices = self.np_random.choice(range(num_tiles), num_key_tiles, replace=False)

            for i in range(num_tiles):
                angle = math.radians(i * angle_step)
                x = self.player_cover_pos[0] + radius * math.cos(angle)
                y = self.player_cover_pos[1] + radius * math.sin(angle)
                is_key = i in key_indices
                self.tiles.append({'pos': (x, y), 'is_key': is_key, 'matched': False, 'radius': 15})

        for i in range(num_cameras):
            angle = self.np_random.uniform(0, 2 * math.pi)
            dist = self.np_random.uniform(180, 220)
            pos = (self.player_cover_pos[0] + dist * math.cos(angle), 
                   self.player_cover_pos[1] + dist * math.sin(angle))
            self.cameras.append({
                'pos': pos, 'angle': self.np_random.uniform(0, 2 * math.pi), 
                'speed': cam_speed * self.np_random.uniform(0.8, 1.2),
                'cone_angle': math.pi / 4, 'cone_length': 150, 'disabled_timer': 0
            })

    def _handle_input(self, movement, space_press):
        if movement != 0 and len(self.tiles) > 1:
            current_pos = self.tiles[self.selected_tile_idx]['pos']
            best_dot = -2
            next_idx = self.selected_tile_idx
            
            move_vec = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]

            for i, tile in enumerate(self.tiles):
                if i == self.selected_tile_idx: continue
                
                direction_vec = (tile['pos'][0] - current_pos[0], tile['pos'][1] - current_pos[1])
                dist = math.hypot(*direction_vec)
                if dist == 0: continue
                
                norm_dir_vec = (direction_vec[0] / dist, direction_vec[1] / dist)
                dot_product = move_vec[0] * norm_dir_vec[0] + move_vec[1] * norm_dir_vec[1]
                
                if dot_product > best_dot:
                    best_dot = dot_product
                    next_idx = i
            
            if best_dot > 0.5:
                self.selected_tile_idx = next_idx
        
        if space_press:
            self._handle_match_attempt()

    def _handle_match_attempt(self):
        if not self.tiles or self.selected_tile_idx >= len(self.tiles):
            return
            
        tile = self.tiles[self.selected_tile_idx]
        if tile['matched'] and not tile.get('is_exit', False):
            return

        is_hit = 0.85 < self.rhythm_timer < 1.0
        
        if self.exit_active and tile.get('is_exit'):
            if is_hit:
                if self.level >= self.FINAL_LEVEL:
                    self.win = True
                else:
                    self._generate_next_level()
                    self.score += 10.0
            else:
                self.player_vulnerable_timer = 30
                self._create_particles(tile['pos'], 20, self.COLOR_DANGER, 2, 4)
                self.score -= 0.1
            return

        if is_hit:
            tile['matched'] = True
            self.score += 1.0
            self.pulses.append({'pos': tile['pos'], 'radius': 10, 'max_radius': 200, 'speed': 8})
            self._create_particles(tile['pos'], 30, self.COLOR_PULSE, 1, 3)
        else:
            self.player_vulnerable_timer = 30
            self._create_particles(tile['pos'], 20, self.COLOR_DANGER, 2, 4)
            self.score -= 0.1

    def _update_game_state(self):
        reward = 0
        for cam in self.cameras:
            if cam['disabled_timer'] > 0:
                cam['disabled_timer'] -= 1
            else:
                cam['angle'] += cam['speed']

        for pulse in self.pulses[:]:
            pulse['radius'] += pulse['speed']
            if pulse['radius'] > pulse['max_radius']:
                self.pulses.remove(pulse)
            else:
                for cam in self.cameras:
                    if cam['disabled_timer'] == 0:
                        dist = math.hypot(pulse['pos'][0] - cam['pos'][0], pulse['pos'][1] - cam['pos'][1])
                        if abs(dist - pulse['radius']) < pulse['speed']:
                            cam['disabled_timer'] = 150
                            reward += 5.0
                            self._create_particles(cam['pos'], 50, self.COLOR_CAMERA_DISABLED, 3, 5)
        
        for p in self.particles[:]:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        if not self.exit_active and self.tiles and all(t['matched'] for t in self.tiles if t['is_key']):
            self.exit_active = True
            exit_tile_found = False
            for i, t in enumerate(self.tiles):
                if t['is_key']:
                    t['is_exit'] = True
                    self.selected_tile_idx = i
                    exit_tile_found = True
                    break
            if not exit_tile_found and self.tiles:
                self.tiles[0]['is_exit'] = True
                self.selected_tile_idx = 0
        return reward

    def _update_timers(self):
        self.rhythm_timer = (self.rhythm_timer + 1.5 / self.FPS) % 1.0
        if self.player_vulnerable_timer > 0:
            self.player_vulnerable_timer -= 1

    def _is_terminated(self):
        if self.win:
            return True
        
        if self.player_vulnerable_timer > 0:
            for cam in self.cameras:
                if cam['disabled_timer'] > 0: continue
                
                dx = self.player_cover_pos[0] - cam['pos'][0]
                dy = self.player_cover_pos[1] - cam['pos'][1]
                dist = math.hypot(dx, dy)
                
                if dist < cam['cone_length']:
                    angle_to_player = math.atan2(dy, dx)
                    angle_diff = (cam['angle'] - angle_to_player + math.pi) % (2 * math.pi) - math.pi
                    if abs(angle_diff) < cam['cone_angle'] / 2:
                        return True
        
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_bg_particles()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level}

    def _render_game(self):
        self._render_pulses()
        self._render_cameras()
        self._render_tiles()
        self._render_particles()

    def _render_bg_particles(self):
        for p in self.bg_particles:
            size = int(p['size'])
            color_val = 30 + p['z'] * 30
            color = (int(color_val*0.5), int(color_val*0.7), color_val)
            pygame.draw.circle(self.screen, color, (int(p['x']), int(p['y'])), size)

    def _update_bg_particles(self):
        for p in self.bg_particles:
            p['y'] += p['z'] * 0.2
            if p['y'] > self.SCREEN_HEIGHT + 10:
                p['y'] = -10
                p['x'] = self.np_random.uniform(0, self.SCREEN_WIDTH)

    def _render_pulses(self):
        for pulse in self.pulses:
            radius = int(pulse['radius'])
            alpha = max(0, 255 * (1 - pulse['radius'] / pulse['max_radius']))
            color = (*self.COLOR_PULSE, int(alpha))
            
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius-1, color)
            pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius-2, color)
            self.screen.blit(temp_surf, (int(pulse['pos'][0]) - radius, int(pulse['pos'][1]) - radius))

    def _render_cameras(self):
        for cam in self.cameras:
            x, y = int(cam['pos'][0]), int(cam['pos'][1])
            is_disabled = cam['disabled_timer'] > 0
            
            if not is_disabled:
                is_detecting = self.player_vulnerable_timer > 0
                cone_color = self.COLOR_DANGER if is_detecting else self.COLOR_CAMERA_CONE
                
                points = [cam['pos']]
                for i in range(-1, 2):
                    angle = cam['angle'] + i * cam['cone_angle'] / 2
                    points.append((x + cam['cone_length'] * math.cos(angle), y + cam['cone_length'] * math.sin(angle)))
                
                temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                pygame.gfxdraw.aapolygon(temp_surf, [(int(p[0]), int(p[1])) for p in points], (*cone_color, 20))
                pygame.gfxdraw.filled_polygon(temp_surf, [(int(p[0]), int(p[1])) for p in points], (*cone_color, 50))
                self.screen.blit(temp_surf, (0, 0))
            
            body_color = self.COLOR_CAMERA_DISABLED if is_disabled else self.COLOR_CAMERA_BODY
            pygame.gfxdraw.filled_circle(self.screen, x, y, 10, body_color)
            pygame.gfxdraw.aacircle(self.screen, x, y, 10, body_color)
            
            lens_x = x + 7 * math.cos(cam['angle'])
            lens_y = y + 7 * math.sin(cam['angle'])
            lens_color = self.COLOR_DANGER if not is_disabled and self.player_vulnerable_timer > 0 else (0,0,0)
            pygame.gfxdraw.filled_circle(self.screen, int(lens_x), int(lens_y), 4, lens_color)

    def _render_tiles(self):
        pulse_val = (math.sin(self.rhythm_timer * 2 * math.pi * 2) + 1) / 2
        
        for i, tile in enumerate(self.tiles):
            x, y = int(tile['pos'][0]), int(tile['pos'][1])
            base_radius = int(tile['radius'])
            
            color = self.COLOR_TILE
            if tile['matched']:
                color = self.COLOR_TILE_MATCHED
                if self.exit_active and tile.get('is_exit'):
                    color = self.COLOR_TILE_EXIT
            elif tile['is_key']:
                color = self.COLOR_TILE_KEY
            
            if i == self.selected_tile_idx:
                sel_radius = base_radius + 4 + 2 * pulse_val
                temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                for j in range(5):
                    alpha = 150 - j * 30
                    pygame.gfxdraw.aacircle(temp_surf, x, y, int(sel_radius + j), (*self.COLOR_TILE_SELECTED, alpha))
                self.screen.blit(temp_surf, (0, 0))

            radius = base_radius + (2 if tile['is_key'] else 0)
            
            temp_glow = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            for j in range(3):
                glow_color = (*color, 70 - j*20)
                pygame.gfxdraw.filled_circle(temp_glow, x, y, radius + 4 - j*2, glow_color)
            self.screen.blit(temp_glow, (0, 0))

            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def _render_particles(self):
        for p in self.particles:
            alpha = 255 * (p['life'] / p['max_life'])
            color = (*p['color'], int(alpha))
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['size'] * (p['life'] / p['max_life']))
            if radius > 0:
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0]-radius, pos[1]-radius))

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        level_text = self.font_large.render(f"DEPTH: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 20, 10))

        bar_w, bar_h = 200, 15
        bar_x, bar_y = self.SCREEN_WIDTH // 2 - bar_w // 2, self.SCREEN_HEIGHT - 40
        pygame.draw.rect(self.screen, self.COLOR_RHYTHM_BAR, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        
        hit_zone_w = bar_w * (1.0 - 0.85)
        hit_zone_x = bar_x + bar_w * 0.85
        pygame.draw.rect(self.screen, self.COLOR_RHYTHM_HIT, (hit_zone_x, bar_y, hit_zone_w, bar_h), border_radius=4)

        cursor_x = bar_x + self.rhythm_timer * bar_w
        pygame.draw.line(self.screen, self.COLOR_PULSE, (cursor_x, bar_y - 2), (cursor_x, bar_y + bar_h + 2), 3)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "FACILITY INFILTRATED" if self.win else "AGENT DETECTED"
            color = self.COLOR_PLAYER if self.win else self.COLOR_DANGER
            end_text = self.font_huge.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, pos, count, color, min_speed, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pos, 'vel': vel, 'life': life, 'max_life': life,
                'color': color, 'size': self.np_random.uniform(2, 5)
            })

    def _generate_bg_particles(self):
        self.bg_particles.clear()
        for _ in range(100):
            self.bg_particles.append({
                'x': self.np_random.uniform(0, self.SCREEN_WIDTH),
                'y': self.np_random.uniform(0, self.SCREEN_HEIGHT),
                'z': self.np_random.uniform(0.1, 1.0),
                'size': self.np_random.uniform(1, 2.5)
            })

if __name__ == '__main__':
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Rhythm Infiltrator")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    action = [0, 0, 0]
    
    running = True
    while running:
        if done:
            print(f"Game Over! Final Score: {info['score']:.2f}, Final Level: {info['level']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                done = True

        keys = pygame.key.get_pressed()
        mov_action = 0
        if keys[pygame.K_UP]: mov_action = 1
        elif keys[pygame.K_DOWN]: mov_action = 2
        elif keys[pygame.K_LEFT]: mov_action = 3
        elif keys[pygame.K_RIGHT]: mov_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        
        current_action = [mov_action, space_action, 0]

        obs, reward, terminated, truncated, info = env.step(current_action)
        done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    pygame.quit()