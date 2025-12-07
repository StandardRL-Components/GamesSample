import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to hit the notes as they reach the yellow markers."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-futuristic rhythm game. Hit the notes on beat to score points and build your combo."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    COLOR_BG = (28, 26, 46) # #1c1a2e
    COLOR_ROAD = (50, 46, 80)
    COLOR_ROAD_LINES = (100, 90, 140)
    COLOR_NOTE_UP = (0, 255, 255) # Cyan
    COLOR_NOTE_DOWN = (255, 0, 255) # Magenta
    COLOR_NOTE_LEFT = (255, 255, 0) # Yellow
    COLOR_NOTE_RIGHT = (0, 255, 128) # Green
    COLOR_MARKER = (255, 200, 0)
    COLOR_HIT = (128, 255, 128)
    COLOR_MISS = (255, 100, 100)
    COLOR_TEXT = (255, 255, 255)
    
    HIT_ZONE_Y = 350
    HIT_TOLERANCE = 25
    
    TOTAL_NOTES = 100
    MAX_MISSES = 5
    MAX_STEPS = 3000 # Generous step limit

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_combo = pygame.font.Font(None, 48)
        self.font_msg = pygame.font.Font(None, 64)
        
        # State variables initialized in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.misses = 0
        self.combo = 0
        self.notes_hit_total = 0
        self.notes_missed_total = 0
        self.notes_hit_for_speedup = 0
        self.note_speed = 0
        self.song_notes = []
        self.note_idx = 0
        self.active_notes = []
        self.particles = []
        self.hit_flashes = []
        self.road_y_offset = 0

        # Map note types to colors and positions
        self.note_lanes = {
            1: {'x': self.SCREEN_WIDTH / 2, 'color': self.COLOR_NOTE_UP},      # Up
            2: {'x': self.SCREEN_WIDTH / 2, 'color': self.COLOR_NOTE_DOWN},     # Down
            3: {'x': self.SCREEN_WIDTH / 2 - 100, 'color': self.COLOR_NOTE_LEFT}, # Left
            4: {'x': self.SCREEN_WIDTH / 2 + 100, 'color': self.COLOR_NOTE_RIGHT} # Right
        }
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.misses = 0
        self.combo = 0
        self.notes_hit_total = 0
        self.notes_missed_total = 0
        self.notes_hit_for_speedup = 0
        self.note_speed = 3.0
        
        self._generate_song()
        self.note_idx = 0

        self.active_notes = []
        self.particles = []
        self.hit_flashes = [0] * 5 # 0:none, 1:U, 2:D, 3:L, 4:R
        self.road_y_offset = 0
        
        return self._get_observation(), self._get_info()

    def _generate_song(self):
        self.song_notes = []
        current_step = 60 # Start first note after 2 seconds at 30fps
        for _ in range(self.TOTAL_NOTES):
            note_type = self.np_random.integers(1, 5) # 1=U, 2=D, 3=L, 4=R
            self.song_notes.append({"spawn_step": current_step, "type": note_type})
            step_interval = self.np_random.integers(15, 30) # 0.5s to 1s between notes
            current_step += step_interval

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        reward = 0
        
        self._spawn_notes()
        miss_penalty = self._update_notes()
        reward -= miss_penalty

        hit_reward = self._handle_action(action)
        reward += hit_reward
        
        self._update_particles()
        self._update_flashes()

        self.road_y_offset = (self.road_y_offset + self.note_speed / 2) % 40

        reward -= len(self.active_notes) * 0.01 # Small penalty for existing notes

        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.notes_hit_total >= self.TOTAL_NOTES * 0.95:
                reward += 50 # Victory bonus
                
        self.steps += 1
        
        # Clamp rewards per spec
        if not terminated:
            reward = np.clip(reward, -10, 10)
        else:
            reward = np.clip(reward, -50, 50)

        obs = self._get_observation()
        info = self._get_info()
        truncated = self.steps >= self.MAX_STEPS

        return (
            obs,
            float(reward),
            terminated or truncated,
            truncated,
            info
        )

    def _spawn_notes(self):
        if self.note_idx < len(self.song_notes) and self.steps >= self.song_notes[self.note_idx]["spawn_step"]:
            note_data = self.song_notes[self.note_idx]
            note_type = note_data['type']
            
            new_note = {
                "type": note_type,
                "y": self.SCREEN_HEIGHT * 0.4, # Vanishing point
                "x": self.note_lanes[note_type]['x'],
                "hit": False
            }
            self.active_notes.append(new_note)
            self.note_idx += 1

    def _update_notes(self):
        miss_penalty = 0
        notes_to_remove = []
        for note in self.active_notes:
            note['y'] += self.note_speed
            if note['y'] > self.HIT_ZONE_Y + self.HIT_TOLERANCE:
                notes_to_remove.append(note)
                self.misses += 1
                self.notes_missed_total += 1
                self.combo = 0
                miss_penalty += 1
                self._create_particles(note['x'], self.HIT_ZONE_Y, self.COLOR_MISS, 20)

        self.active_notes = [n for n in self.active_notes if n not in notes_to_remove]
        return miss_penalty

    def _handle_action(self, action):
        movement = action[0]
        hit_reward = 0
        
        if movement == 0:
            return 0
        
        self.hit_flashes[movement] = 10 # Flash duration in frames

        best_note = None
        min_dist = float('inf')

        for note in self.active_notes:
            if note['type'] == movement:
                dist = abs(note['y'] - self.HIT_ZONE_Y)
                if dist < self.HIT_TOLERANCE and dist < min_dist:
                    min_dist = dist
                    best_note = note
        
        if best_note:
            best_note['hit'] = True
            self.active_notes.remove(best_note)
            
            # --- Score and Combo ---
            self.score += 10 * (1 + self.combo // 10)
            self.combo += 1
            self.notes_hit_total += 1
            self.notes_hit_for_speedup += 1
            
            # --- Rewards ---
            hit_reward += 1 # Base reward for hit
            if self.combo > 0 and self.combo % 10 == 0:
                hit_reward += 5 # Combo bonus
            
            # --- Effects ---
            self._create_particles(best_note['x'], self.HIT_ZONE_Y, self.note_lanes[best_note['type']]['color'], 30)
            
            # --- Difficulty Scaling ---
            if self.notes_hit_for_speedup >= 20:
                self.note_speed = min(8.0, self.note_speed + 0.5)
                self.notes_hit_for_speedup = 0
        
        return hit_reward

    def _check_termination(self):
        return (
            self.misses >= self.MAX_MISSES or
            (self.notes_hit_total + self.notes_missed_total >= self.TOTAL_NOTES)
        )

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': [x, y], 'vel': vel, 'color': color, 'lifespan': lifespan, 'max_life': lifespan})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        
    def _update_flashes(self):
        for i in range(len(self.hit_flashes)):
            self.hit_flashes[i] = max(0, self.hit_flashes[i] - 1)
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_road()
        self._render_hit_zone()
        self._render_notes()
        self._render_particles()

    def _render_road(self):
        vp_x, vp_y = self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.4
        
        # Road edges
        poly = [
            (vp_x - 5, vp_y), (vp_x + 5, vp_y),
            (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), (0, self.SCREEN_HEIGHT)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, poly, self.COLOR_ROAD)

        # Lane lines
        lanes_x = [self.SCREEN_WIDTH / 2 - 100, self.SCREEN_WIDTH / 2, self.SCREEN_WIDTH / 2 + 100]
        for lane_x in lanes_x:
            # Project lane center to vanishing point
            bottom_x_offset = (lane_x - vp_x)
            bottom_x = vp_x + bottom_x_offset * (self.SCREEN_HEIGHT / vp_y)
            pygame.draw.line(self.screen, self.COLOR_ROAD_LINES, (vp_x, vp_y), (bottom_x, self.SCREEN_HEIGHT), 2)
        
        # Horizontal lines for motion
        for i in range(10):
            y = vp_y + (i * 40 + self.road_y_offset)
            if y > self.SCREEN_HEIGHT: continue
            
            persp = (y - vp_y) / (self.SCREEN_HEIGHT - vp_y)
            width = self.SCREEN_WIDTH * persp
            start_x = vp_x - width / 2
            end_x = vp_x + width / 2
            
            if y > vp_y:
                pygame.draw.line(self.screen, self.COLOR_ROAD_LINES, (start_x, y), (end_x, y), 1)

    def _render_hit_zone(self):
        note_map = { 1: "↑", 2: "↓", 3: "←", 4: "→" }
        for i in range(1, 5):
            lane_info = self.note_lanes[i]
            x, color = lane_info['x'], lane_info['color']
            
            # Special handling for up/down which share a lane
            y_pos = self.HIT_ZONE_Y
            if i == 1: y_pos -= 15
            if i == 2: y_pos += 15

            flash_alpha = int(255 * (self.hit_flashes[i] / 10))
            if flash_alpha > 0:
                flash_surf = pygame.Surface((60, 30), pygame.SRCALPHA)
                flash_color = (*self.COLOR_MARKER, flash_alpha)
                pygame.draw.rect(flash_surf, flash_color, flash_surf.get_rect(), border_radius=8)
                self.screen.blit(flash_surf, (int(x - 30), int(y_pos - 15)))

            # Draw the marker symbol
            text = self.font_combo.render(note_map[i], True, self.COLOR_MARKER)
            text_rect = text.get_rect(center=(int(x), int(y_pos)))
            self.screen.blit(text, text_rect)
            
    def _render_notes(self):
        vp_y = self.SCREEN_HEIGHT * 0.4
        note_map = { 1: "↑", 2: "↓", 3: "←", 4: "→" }
        # Draw notes from back to front
        for note in sorted(self.active_notes, key=lambda n: n['y']):
            y = note['y']
            persp = max(0, (y - vp_y) / (self.HIT_ZONE_Y - vp_y))
            size = int(10 + 30 * persp)
            
            x = note['x']
            color = self.note_lanes[note['type']]['color']

            # Glow effect
            glow_radius = int(size * 0.7)
            if glow_radius > 0:
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                # FIX: filled_circle takes 5 arguments: surface, x, y, radius, color
                pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*color, 50))
                self.screen.blit(glow_surf, (int(x - glow_radius), int(y - glow_radius)))
            
            # Note symbol
            font_size = int(12 + 30 * persp)
            if font_size > 0:
                note_font = pygame.font.Font(None, font_size)
                text = note_font.render(note_map[note['type']], True, color)
                text_rect = text.get_rect(center=(int(x), int(y)))
                self.screen.blit(text, text_rect)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_life']))
            color_with_alpha = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['lifespan'] / p['max_life'] * 4)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color_with_alpha)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Combo
        if self.combo > 1:
            combo_text = self.font_combo.render(f"{self.combo}x", True, self.COLOR_TEXT)
            text_rect = combo_text.get_rect(midtop=(self.SCREEN_WIDTH / 2, 10))
            self.screen.blit(combo_text, text_rect)

        # Misses
        miss_text = self.font_ui.render(f"MISSES: {self.misses}/{self.MAX_MISSES}", True, self.COLOR_MISS)
        text_rect = miss_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(miss_text, text_rect)

        # Game Over Message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            msg = "CLEAR!" if self.notes_hit_total >= self.TOTAL_NOTES * 0.95 else "FAILED"
            color = self.COLOR_HIT if self.notes_hit_total >= self.TOTAL_NOTES * 0.95 else self.COLOR_MISS
            
            msg_text = self.font_msg.render(msg, True, color)
            text_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "combo": self.combo,
            "misses": self.misses,
            "notes_hit": self.notes_hit_total,
            "notes_total": self.TOTAL_NOTES
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's a demonstration of how to use the environment
    # Make sure to re-enable the normal video driver
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Rhythm Highway")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                elif event.key == pygame.K_ESCAPE:
                    running = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()