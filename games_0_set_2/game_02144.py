
# Generated: 2025-08-27T19:23:36.483820
# Source Brief: brief_02144.md
# Brief Index: 2144

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Use arrows to move the cursor. Press space to select a gem, "
        "then move to an adjacent gem and press space again to swap. "
        "Clear gems by matching 3 or more of the same color. "
        "Hold shift to restart the current stage."
    )

    game_description = (
        "A fast-paced match-3 puzzle game. Race against the clock to score points "
        "by matching gems and clear all three stages to win."
    )

    auto_advance = True
    
    # --- Game Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_GEM_TYPES = 6
    TARGET_SCORE = 100
    TOTAL_STAGES = 3
    TIME_PER_STAGE_SECONDS = 60
    
    # --- Visual Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_AREA_WIDTH = 320
    GEM_SIZE = GRID_AREA_WIDTH // GRID_WIDTH
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_AREA_WIDTH) // 2
    
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    COLOR_CURSOR = (255, 255, 100)
    COLOR_SELECTED = (255, 255, 255, 150)
    
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]

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
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        self.game_state = "IDLE"
        self.grid = []
        self.animations = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_gem_pos = None
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.timer = 0
        self.game_over = False
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.move_cooldown = 0
        
        self.reward_buffer = 0
        
        self.reset()
        self.validate_implementation()

    def _reset_stage(self):
        self.score = 0
        self.timer = self.TIME_PER_STAGE_SECONDS * 30  # 30 FPS
        self.game_state = "IDLE"
        self.animations = []
        self.particles = []
        self.selected_gem_pos = None
        self._init_grid()
        while self._find_all_matches():
            self._init_grid() # Ensure no starting matches

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.stage = 1
        self.game_over = False
        self.reward_buffer = 0
        
        self._reset_stage()
        
        return self._get_observation(), self._get_info()

    def _init_grid(self):
        self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_WIDTH, self.GRID_HEIGHT))

    def step(self, action):
        reward = 0
        terminated = False
        
        if not self.game_over:
            self._handle_input(action)
            self._update_game_logic()
        
        if self.game_state == "IDLE" and self.reward_buffer != 0:
            reward += self.reward_buffer
            self.reward_buffer = 0

        # Check for stage clear
        if self.score >= self.TARGET_SCORE and self.game_state == "IDLE":
            if self.stage < self.TOTAL_STAGES:
                self.stage += 1
                self._reset_stage()
                reward += 1.0 # Stage clear reward
                self.game_state = "STAGE_CLEAR"
                self.animations.append({"type": "message", "text": f"STAGE {self.stage}", "timer": 60})
            else:
                self.game_over = True
                terminated = True
                reward = 100.0 # Win game reward
                self.game_state = "GAME_WON"
                self.animations.append({"type": "message", "text": "YOU WIN!", "timer": 180})

        # Check for time out
        if self.timer <= 0 and not self.game_over:
            self.game_over = True
            terminated = True
            reward = -100.0 # Lose game reward
            self.game_state = "GAME_OVER"
            self.animations.append({"type": "message", "text": "GAME OVER", "timer": 180})

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        if self.game_state != "IDLE":
            return
            
        if shift_press:
            self._reset_stage()
            return

        # Handle cursor movement with cooldown
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        elif movement != 0:
            self.move_cooldown = 4 # frames
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)
        
        if space_press:
            if self.selected_gem_pos is None:
                self.selected_gem_pos = list(self.cursor_pos)
                # SFX: select_gem.wav
            else:
                dist = abs(self.selected_gem_pos[0] - self.cursor_pos[0]) + abs(self.selected_gem_pos[1] - self.cursor_pos[1])
                if dist == 1: # Is adjacent
                    self._start_swap(self.selected_gem_pos, self.cursor_pos)
                else: # Not adjacent, change selection
                    self.selected_gem_pos = list(self.cursor_pos)
                    # SFX: select_gem.wav
    
    def _start_swap(self, pos1, pos2):
        self.game_state = "SWAPPING"
        self.animations.append({
            "type": "swap", "pos1": pos1, "pos2": pos2, "progress": 0, "duration": 10
        })
        self.selected_gem_pos = None
        # SFX: swap.wav

    def _update_game_logic(self):
        self.timer = max(0, self.timer - 1)
        self._update_animations()
        self._update_particles()
        
        if self.game_state == "CHECK_MATCHES":
            self._find_and_process_matches()

    def _find_and_process_matches(self):
        matches = self._find_all_matches()
        if matches:
            num_gems_matched = len(matches)
            self.reward_buffer += 0.1 * num_gems_matched
            self.score += num_gems_matched
            
            # SFX: match.wav
            for x, y in matches:
                self._create_particles(x, y, self.grid[x][y])
                self.grid[x][y] = -1 # Mark for removal
            
            self.game_state = "APPLY_GRAVITY"
        else:
            self.game_state = "IDLE"

    def _apply_gravity(self):
        fall_animations = []
        for x in range(self.GRID_WIDTH):
            write_y = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x][y] != -1:
                    if y != write_y:
                        fall_animations.append({"type": "fall", "from_pos": [x, y], "to_pos": [x, write_y], "progress": 0, "duration": 15})
                        self.grid[x][write_y] = self.grid[x][y]
                        self.grid[x][y] = -1
                    write_y -= 1
        
        if fall_animations:
            self.animations.extend(fall_animations)
            self.game_state = "FALLING"
            # SFX: gems_fall.wav
        else:
            self._refill_grid()

    def _refill_grid(self):
        refill_animations = []
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x][y] == -1:
                    self.grid[x][y] = self.np_random.integers(0, self.NUM_GEM_TYPES)
                    refill_animations.append({"type": "refill", "pos": [x, y], "progress": 0, "duration": 10})
        
        if refill_animations:
            self.animations.extend(refill_animations)
            self.game_state = "REFILLING"
        else:
            self.game_state = "CHECK_MATCHES"

    def _find_all_matches(self):
        matches = set()
        # Horizontal
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if self.grid[x][y] != -1 and self.grid[x][y] == self.grid[x+1][y] == self.grid[x+2][y]:
                    matches.add((x, y)); matches.add((x+1, y)); matches.add((x+2, y))
        # Vertical
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if self.grid[x][y] != -1 and self.grid[x][y] == self.grid[x][y+1] == self.grid[x][y+2]:
                    matches.add((x, y)); matches.add((x, y+1)); matches.add((x, y+2))
        return matches

    def _update_animations(self):
        if not self.animations:
            if self.game_state == "APPLY_GRAVITY": self._apply_gravity()
            if self.game_state == "REFILLING": self._refill_grid()
            return
            
        finished_animations = []
        for anim in self.animations:
            anim["progress"] += 1
            if anim["progress"] >= anim.get("duration", 60):
                finished_animations.append(anim)

        for anim in finished_animations:
            self.animations.remove(anim)
            if anim["type"] == "swap":
                x1, y1 = anim["pos1"]; x2, y2 = anim["pos2"]
                self.grid[x1][y1], self.grid[x2][y2] = self.grid[x2][y2], self.grid[x1][y1]
                self.game_state = "CHECK_MATCHES"
            elif anim["type"] == "message":
                if self.game_state == "STAGE_CLEAR":
                    self.game_state = "IDLE"
            # Fall and refill animations just run, their completion is checked by `if not self.animations`
        
        if not self.animations:
            if self.game_state == "FALLING": self._refill_grid()
            elif self.game_state == "REFILLING": self.game_state = "CHECK_MATCHES"


    def _create_particles(self, grid_x, grid_y, gem_type):
        px, py = self._grid_to_pixel(grid_x, grid_y)
        px += self.GEM_SIZE // 2
        py += self.GEM_SIZE // 2
        color = self.GEM_COLORS[gem_type]
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": [px, py], "vel": vel, "life": random.randint(15, 30),
                "size": random.uniform(2, 5), "color": color
            })
            
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage}
        
    def _grid_to_pixel(self, x, y):
        return self.GRID_OFFSET_X + x * self.GEM_SIZE, self.GRID_OFFSET_Y + y * self.GEM_SIZE

    def _render_text(self, text, font, color, pos, shadow_color=None, shadow_offset=2):
        text_surf = font.render(str(text), True, color)
        text_rect = text_surf.get_rect(center=pos)
        if shadow_color:
            shadow_surf = font.render(str(text), True, shadow_color)
            shadow_rect = shadow_surf.get_rect(center=(pos[0] + shadow_offset, pos[1] + shadow_offset))
            self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_AREA_WIDTH, self.GRID_AREA_WIDTH)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=5)
        
        # Prepare animated gem data
        animating_gems = {} # (x,y) -> (pixel_x, pixel_y, gem_type)
        for anim in self.animations:
            if anim["type"] == "swap":
                p = anim["progress"] / anim["duration"]
                x1, y1 = anim["pos1"]; x2, y2 = anim["pos2"]
                px1, py1 = self._grid_to_pixel(x1, y1); px2, py2 = self._grid_to_pixel(x2, y2)
                
                animating_gems[(x1,y1)] = (px1 + (px2-px1)*p, py1 + (py2-py1)*p, self.grid[x2][y2])
                animating_gems[(x2,y2)] = (px2 + (px1-px2)*p, py2 + (py1-py2)*p, self.grid[x1][y1])
            elif anim["type"] == "fall":
                p = anim["progress"] / anim["duration"]
                p = 1 - (1-p)**2 # Ease in
                x1, y1 = anim["from_pos"]; x2, y2 = anim["to_pos"]
                px1, py1 = self._grid_to_pixel(x1, y1); px2, py2 = self._grid_to_pixel(x2, y2)
                animating_gems[(x1,y1)] = (px1, py1 + (py2-py1)*p, self.grid[x2][y2])
            elif anim["type"] == "refill":
                p = anim["progress"] / anim["duration"]
                x, y = anim["pos"]
                px, py = self._grid_to_pixel(x, y)
                animating_gems[(x,y)] = (px, py - (1-p) * self.GEM_SIZE, self.grid[x][y])

        # Draw gems
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x][y] == -1: continue
                
                if (x,y) in animating_gems:
                    px, py, gem_type = animating_gems[(x,y)]
                else:
                    px, py = self._grid_to_pixel(x, y)
                    gem_type = self.grid[x][y]
                
                self._draw_gem(px, py, gem_type)

        # Draw cursor and selection
        if self.game_state == "IDLE":
            cx, cy = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
            cursor_rect = pygame.Rect(cx, cy, self.GEM_SIZE, self.GEM_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=6)
            
            if self.selected_gem_pos:
                sx, sy = self._grid_to_pixel(self.selected_gem_pos[0], self.selected_gem_pos[1])
                s = pygame.Surface((self.GEM_SIZE, self.GEM_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(s, self.COLOR_SELECTED, (0,0,self.GEM_SIZE,self.GEM_SIZE), border_radius=6)
                self.screen.blit(s, (sx, sy))

        self._render_particles()
        self._render_messages()

    def _draw_gem(self, px, py, gem_type):
        color = self.GEM_COLORS[gem_type]
        rect = pygame.Rect(int(px), int(py), self.GEM_SIZE, self.GEM_SIZE)
        
        # Simple geometric shapes
        shape_padding = int(self.GEM_SIZE * 0.15)
        shape_rect = rect.inflate(-shape_padding*2, -shape_padding*2)
        
        highlight_color = tuple(min(255, c + 60) for c in color)
        shadow_color = tuple(max(0, c - 60) for c in color)

        if gem_type == 0: # Circle
            pygame.gfxdraw.filled_circle(self.screen, shape_rect.centerx, shape_rect.centery, shape_rect.width//2, color)
            pygame.gfxdraw.aacircle(self.screen, shape_rect.centerx, shape_rect.centery, shape_rect.width//2, highlight_color)
        elif gem_type == 1: # Square
            pygame.draw.rect(self.screen, color, shape_rect, border_radius=3)
            pygame.draw.rect(self.screen, highlight_color, shape_rect, 1, border_radius=3)
        elif gem_type == 2: # Diamond
            points = [shape_rect.midtop, shape_rect.midright, shape_rect.midbottom, shape_rect.midleft]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, highlight_color)
        elif gem_type == 3: # Triangle
            points = [shape_rect.midtop, shape_rect.bottomright, shape_rect.bottomleft]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, highlight_color)
        elif gem_type == 4: # Hexagon
            s = shape_rect.width / 2
            h = s * math.sqrt(3)/2
            cx, cy = shape_rect.center
            points = [
                (cx, cy-s), (cx+h, cy-s/2), (cx+h, cy+s/2),
                (cx, cy+s), (cx-h, cy+s/2), (cx-h, cy-s/2)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, highlight_color)
        else: # Star (simple)
            s = shape_rect.width / 2
            cx, cy = shape_rect.center
            points = []
            for i in range(10):
                r = s if i % 2 == 0 else s * 0.5
                angle = i * math.pi / 5 - math.pi/2
                points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, highlight_color)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color = p["color"] + (alpha,)
            s = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(s, (p["pos"][0] - p["size"], p["pos"][1] - p["size"]))

    def _render_messages(self):
        for anim in self.animations:
            if anim["type"] == "message":
                self._render_text(anim["text"], self.font_large, self.COLOR_TEXT, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2), self.COLOR_TEXT_SHADOW)

    def _render_ui(self):
        # Score
        self._render_text(f"Score: {self.score}", self.font_medium, self.COLOR_TEXT, (100, 30), self.COLOR_TEXT_SHADOW)
        # Timer
        time_left = math.ceil(self.timer / 30)
        timer_color = (255, 100, 100) if time_left <= 10 else self.COLOR_TEXT
        self._render_text(f"Time: {time_left}", self.font_medium, timer_color, (self.SCREEN_WIDTH - 100, 30), self.COLOR_TEXT_SHADOW)
        # Stage
        self._render_text(f"Stage: {self.stage} / {self.TOTAL_STAGES}", self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 30), self.COLOR_TEXT_SHADOW)

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It is not part of the Gymnasium environment itself
    # but is useful for testing and demonstration.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Gem Matcher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Player Input ---
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total: {total_reward:.2f}, Score: {info['score']}")
            
        if terminated or truncated:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # In a real scenario, you'd reset here. We'll just let the end-game message display.
            # obs, info = env.reset()
            # total_reward = 0
            
        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Match the environment's internal FPS
        
    pygame.quit()