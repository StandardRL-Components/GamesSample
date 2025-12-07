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
        "Use Left/Down/Up/Right arrow keys to select a lane. Press Space to hit the notes in the selected lane."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced rhythm game. Hit the falling notes on the beat to build your combo and clear all three stages."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.BPM = 120.0
        self.BEATS_PER_SEC = self.BPM / 60.0
        self.LANE_COUNT = 4
        self.TOTAL_GAME_DURATION_S = 180 # 3 stages * 60s
        self.TOTAL_GAME_STEPS = self.TOTAL_GAME_DURATION_S * self.FPS
        self.STAGE_DURATION_STEPS = (self.TOTAL_GAME_DURATION_S // 3) * self.FPS
        self.MAX_MISSES = 5
        self.NOTE_SPEED = self.HEIGHT / (2.0 * self.FPS) # Note is on screen for 2 seconds
        self.TARGET_Y = self.HEIGHT - 50
        self.HIT_WINDOW_PERFECT = 3 * self.NOTE_SPEED # +/- 3 frames from target
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_combo = pygame.font.Font(None, 48)
        self.font_feedback = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG_START = (10, 5, 20)
        self.COLOR_BG_END = (30, 10, 40)
        self.COLOR_TARGET_LINE = (200, 200, 255)
        self.COLOR_LANE_GUIDE = (255, 255, 255, 30)
        self.COLOR_LANE_ACTIVE = (255, 255, 255, 90)
        self.NOTE_COLORS = [
            (50, 150, 255),  # Blue
            (50, 255, 150),  # Green
            (255, 200, 50),  # Yellow
            (255, 80, 80),   # Red
        ]
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_MISS = (100, 100, 100)

        # Initialize state variables
        self.np_random = None
        self.note_chart = []
        self.notes = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.combo = 0
        self.max_combo = 0
        self.hits = 0
        self.misses = 0
        self.total_notes_in_chart = 0
        # Initialize last_action to a valid default value before validation
        self.last_action = np.array([0, 0, 0])

        # Defer reset until after validation is complete
        # self.validate_implementation()
        self.reset()
    
    def _generate_note_chart(self):
        chart = []
        steps_per_beat = self.FPS / self.BEATS_PER_SEC
        
        # Stage 1: Simple, single notes
        min_beat_sep1, max_beat_sep1 = 2.0, 4.0
        current_beat = 4.0
        while current_beat * steps_per_beat < self.STAGE_DURATION_STEPS:
            lane = self.np_random.integers(0, self.LANE_COUNT)
            chart.append((int(current_beat * steps_per_beat), lane))
            current_beat += self.np_random.uniform(min_beat_sep1, max_beat_sep1) * 0.5

        # Stage 2: Faster notes, some doubles
        min_beat_sep2, max_beat_sep2 = 1.0, 3.0
        current_beat = (self.STAGE_DURATION_STEPS / steps_per_beat)
        while current_beat * steps_per_beat < 2 * self.STAGE_DURATION_STEPS:
            num_notes = 1 if self.np_random.random() > 0.2 else 2 # 20% chance of a double
            lanes = self.np_random.choice(self.LANE_COUNT, size=num_notes, replace=False)
            for lane in lanes:
                chart.append((int(current_beat * steps_per_beat), lane))
            current_beat += self.np_random.uniform(min_beat_sep2, max_beat_sep2) * 0.5

        # Stage 3: Very fast, doubles and triples
        min_beat_sep3, max_beat_sep3 = 0.5, 2.0
        current_beat = (2 * self.STAGE_DURATION_STEPS / steps_per_beat)
        while current_beat * steps_per_beat < self.TOTAL_GAME_STEPS:
            p = self.np_random.random()
            if p > 0.6: num_notes = 1 # 40% single
            elif p > 0.2: num_notes = 2 # 40% double
            else: num_notes = 3 # 20% triple
            lanes = self.np_random.choice(self.LANE_COUNT, size=num_notes, replace=False)
            for lane in lanes:
                chart.append((int(current_beat * steps_per_beat), lane))
            current_beat += self.np_random.uniform(min_beat_sep3, max_beat_sep3) * 0.5
            
        return sorted(chart)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # self.np_random is now seeded and available from super().reset()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.combo = 0
        self.max_combo = 0
        self.hits = 0
        self.misses = 0
        self.notes = []
        self.particles = []
        self.note_chart = self._generate_note_chart()
        self.total_notes_in_chart = len(self.note_chart)
        self.last_action = np.array([0, 0, 0]) # No-op
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.last_action = action
        reward = 0
        
        if not self.game_over:
            self.steps += 1
            
            # --- Game Logic ---
            # The reward from _update_notes is local and used for fine-tuning RL agents.
            # The main step reward is for major game events.
            note_reward = self._update_notes(action)
            reward += note_reward # Accumulate rewards

            self._spawn_notes()
            self._update_particles()
            
            # Check for stage completion rewards
            if self.steps == self.STAGE_DURATION_STEPS or self.steps == 2 * self.STAGE_DURATION_STEPS:
                reward += 100

            # --- Termination ---
            terminated = (self.misses >= self.MAX_MISSES) or (self.steps >= self.TOTAL_GAME_STEPS)
            if terminated and not self.game_over:
                self.game_over = True
                if self.misses >= self.MAX_MISSES:
                    reward -= 100
                elif self.steps >= self.TOTAL_GAME_STEPS:
                    reward += 100 # Final stage completion
                    if self.hits == self.total_notes_in_chart and self.total_notes_in_chart > 0:
                        reward += 1000 # Perfect game bonus
        else:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False, # Truncated is not used in this environment
            self._get_info()
        )

    def _spawn_notes(self):
        while self.note_chart and self.steps >= self.note_chart[0][0]:
            _, lane = self.note_chart.pop(0)
            new_note = {
                "y": 0,
                "lane": lane,
                "color": self.NOTE_COLORS[lane % len(self.NOTE_COLORS)],
                "active": True
            }
            self.notes.append(new_note)

    def _update_notes(self, action):
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1

        lane_map = {3: 0, 2: 1, 1: 2, 4: 3} # left, down, up, right
        targeted_lane = lane_map.get(movement, -1)
        
        reward = 0
        
        if space_pressed and targeted_lane != -1:
            notes_in_lane = [n for n in self.notes if n["lane"] == targeted_lane and n["active"]]
            if notes_in_lane:
                # Find closest note to target line
                closest_note = min(notes_in_lane, key=lambda n: abs(n["y"] - self.TARGET_Y))
                
                # Check for hit
                if abs(closest_note["y"] - self.TARGET_Y) < self.HIT_WINDOW_PERFECT:
                    closest_note["active"] = False
                    self.hits += 1
                    self.combo += 1
                    self.max_combo = max(self.max_combo, self.combo)
                    self.score += 10 * (1 + self.combo // 10)
                    reward += 1
                    if self.combo > 0 and self.combo % 10 == 0:
                        reward += 5
                    self._create_hit_particles(targeted_lane, closest_note["color"])
        
        # Move notes and check for misses
        for note in self.notes[:]:
            if not note["active"]:
                self.notes.remove(note)
                continue
            
            note["y"] += self.NOTE_SPEED
            
            if note["y"] > self.TARGET_Y + self.HIT_WINDOW_PERFECT:
                note["active"] = False
                self.misses += 1
                self.combo = 0
                reward -= 1
                self._create_miss_particles(note["lane"])
        
        return reward

    def _create_hit_particles(self, lane, color):
        x = (lane + 0.5) * (self.WIDTH / self.LANE_COUNT)
        y = self.TARGET_Y
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            particle = {
                "x": x, "y": y,
                "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "radius": self.np_random.uniform(3, 7),
                "color": color,
                "lifespan": self.FPS // 2 # 0.5 seconds
            }
            self.particles.append(particle)

    def _create_miss_particles(self, lane):
        x = (lane + 0.5) * (self.WIDTH / self.LANE_COUNT)
        y = self.TARGET_Y
        for _ in range(10):
            angle = self.np_random.uniform(math.pi, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            particle = {
                "x": x, "y": y,
                "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "radius": self.np_random.uniform(2, 4),
                "color": self.COLOR_MISS,
                "lifespan": self.FPS // 3 # 0.33 seconds
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.1 # Gravity
            p["lifespan"] -= 1
            p["radius"] *= 0.95
            if p["lifespan"] <= 0 or p["radius"] < 0.5:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        self.screen.fill(self.COLOR_BG_START)
        # Gradient effect is too slow, using solid color for performance
        # for y in range(self.HEIGHT):
        #     r = self.COLOR_BG_START[0] + (self.COLOR_BG_END[0] - self.COLOR_BG_START[0]) * y / self.HEIGHT
        #     g = self.COLOR_BG_START[1] + (self.COLOR_BG_END[1] - self.COLOR_BG_START[1]) * y / self.HEIGHT
        #     b = self.COLOR_BG_START[2] + (self.COLOR_BG_END[2] - self.COLOR_BG_START[2]) * y / self.HEIGHT
        #     pygame.draw.line(self.screen, (r, g, b), (0, y), (self.WIDTH, y))

    def _render_game(self):
        lane_width = self.WIDTH / self.LANE_COUNT
        
        # Draw lane guides and active lanes
        movement = self.last_action[0]
        lane_map = {3: 0, 2: 1, 1: 2, 4: 3}
        targeted_lane = lane_map.get(movement, -1)
        
        for i in range(1, self.LANE_COUNT):
            x = i * lane_width
            pygame.gfxdraw.vline(self.screen, int(x), 0, self.HEIGHT, self.COLOR_LANE_GUIDE)
        
        if targeted_lane != -1 and not self.game_over:
            rect = pygame.Rect(targeted_lane * lane_width, 0, lane_width, self.HEIGHT)
            s = pygame.Surface((lane_width, self.HEIGHT), pygame.SRCALPHA)
            s.fill(self.COLOR_LANE_ACTIVE)
            self.screen.blit(s, (rect.x, rect.y))

        # Draw target line with pulse
        pulse = abs(math.sin(self.steps * self.BEATS_PER_SEC * math.pi / self.FPS))
        alpha = 150 + 105 * pulse
        pygame.gfxdraw.hline(self.screen, 0, self.WIDTH, self.TARGET_Y, (*self.COLOR_TARGET_LINE, int(alpha)))
        pygame.gfxdraw.hline(self.screen, 0, self.WIDTH, self.TARGET_Y - 1, (*self.COLOR_TARGET_LINE, int(alpha/2)))
        pygame.gfxdraw.hline(self.screen, 0, self.WIDTH, self.TARGET_Y + 1, (*self.COLOR_TARGET_LINE, int(alpha/2)))
        
        # Draw notes
        note_width = lane_width * 0.8
        note_height = self.NOTE_SPEED * 2.5
        for note in self.notes:
            if not note["active"]: continue
            x = (note["lane"] + 0.5) * lane_width
            y = note["y"]
            rect = pygame.Rect(x - note_width/2, y - note_height/2, note_width, note_height)
            pygame.draw.rect(self.screen, note["color"], rect, border_radius=5)
            pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in note["color"]), rect, width=2, border_radius=5)
            
        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p["x"]), int(p["y"]), int(p["radius"]), p["color"])

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Accuracy
        total_judged = self.hits + self.misses
        accuracy = (self.hits / total_judged * 100) if total_judged > 0 else 100.0
        acc_text = self.font_ui.render(f"Acc: {accuracy:.2f}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(acc_text, (self.WIDTH - acc_text.get_width() - 10, 10))
        
        # Misses
        miss_text = self.font_ui.render(f"Miss: {self.misses}/{self.MAX_MISSES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(miss_text, (self.WIDTH - miss_text.get_width() - 10, 40))
        
        # Stage
        current_stage = min(3, self.steps // self.STAGE_DURATION_STEPS + 1)
        stage_text = self.font_ui.render(f"Stage: {current_stage}/3", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (10, 40))

        # Combo
        if self.combo > 2:
            combo_text = self.font_combo.render(f"{self.combo}", True, self.COLOR_UI_TEXT)
            combo_label = self.font_feedback.render("COMBO", True, self.COLOR_UI_TEXT)
            self.screen.blit(combo_text, (self.WIDTH/2 - combo_text.get_width()/2, self.HEIGHT/2 - 50))
            self.screen.blit(combo_label, (self.WIDTH/2 - combo_label.get_width()/2, self.HEIGHT/2))

        # Game Over Message
        if self.game_over:
            result_text_str = ""
            if self.misses >= self.MAX_MISSES:
                result_text_str = "FAILED"
                color = (255, 50, 50)
            else:
                result_text_str = "CLEAR!"
                color = (50, 255, 50)
            
            result_text = self.font_game_over.render(result_text_str, True, color)
            self.screen.blit(result_text, (self.WIDTH/2 - result_text.get_width()/2, self.HEIGHT/2 - 60))
            
            final_score_text = self.font_ui.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            self.screen.blit(final_score_text, (self.WIDTH/2 - final_score_text.get_width()/2, self.HEIGHT/2 + 20))
            
            max_combo_text = self.font_ui.render(f"Max Combo: {self.max_combo}", True, self.COLOR_UI_TEXT)
            self.screen.blit(max_combo_text, (self.WIDTH/2 - max_combo_text.get_width()/2, self.HEIGHT/2 + 50))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "combo": self.combo,
            "max_combo": self.max_combo,
            "hits": self.hits,
            "misses": self.misses,
            "stage": min(3, self.steps // self.STAGE_DURATION_STEPS + 1) if not self.game_over else 3
        }

    def close(self):
        pygame.quit()