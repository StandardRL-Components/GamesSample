import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:07:43.823247
# Source Brief: brief_02591.md
# Brief Index: 2591
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent must stabilize a glitching virtual world.
    The agent selects debug tools and applies them with the correct rhythm to fix
    glitches, earning points and eventually creating powerful code snippets.
    """
    metadata = {"render_modes": ["rgb_array"]}

    auto_advance = False
    game_description = (
        "Stabilize a glitching virtual world by selecting debug tools and applying them "
        "with the correct rhythm to fix anomalies and create powerful code snippets."
    )
    user_guide = (
        "Controls: Use ↑↓ to select a tool and ←→ to select a code snippet. "
        "Press space or shift to apply the selected item to the oldest glitch, matching its rhythm."
    )

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_ROWS = 8
    GRID_COLS = 12
    CELL_SIZE = 40
    UI_HEIGHT = 80
    GAME_AREA_HEIGHT = SCREEN_HEIGHT - UI_HEIGHT
    MAX_STEPS = 2000
    MAX_GLITCH_LEVEL = 100.0

    # --- Colors (Cyberpunk Palette) ---
    COLOR_BG = (10, 10, 30)
    COLOR_GRID = (30, 50, 90)
    COLOR_STABLE = (0, 255, 150)
    COLOR_UNSTABLE = (255, 50, 80)
    COLOR_TEXT = (220, 220, 255)
    COLOR_TEXT_SHADOW = (20, 20, 40)
    COLOR_SNIPPET = (255, 200, 0)

    TOOL_COLORS = [
        (0, 150, 255),  # Blue - Type 0
        (255, 0, 255),  # Magenta - Type 1
        (50, 255, 50),   # Green - Type 2
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.glitch_level = 0.0
        self.glitch_spawn_rate = 0.0
        self.glitches = []
        self.particles = []
        self.tools = []
        self.snippets = []
        self.selected_tool_idx = 0
        self.selected_snippet_idx = 0
        self.last_fix_chain = []
        self.known_snippet_patterns = {}
        self.action_feedback = [] # For visualizing actions


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.glitch_level = 20.0
        self.glitch_spawn_rate = 0.02
        self.glitches = []
        self.particles = []
        self.action_feedback = []
        
        # --- Tools & Snippets ---
        self.tools = [
            {"name": "Defragger", "type": 0, "color": self.TOOL_COLORS[0]},
        ]
        self.snippets = []
        self.selected_tool_idx = 0
        self.selected_snippet_idx = -1 # -1 indicates no snippet selected
        self.last_fix_chain = []
        self.known_snippet_patterns = {
            "Auto-Defrag": (0, 0),
            "Packet-Purge": (1, 1),
            "System-Flush": (0, 1, 0)
        }

        # --- Initial Game Setup ---
        for _ in range(3):
            self._spawn_glitch()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Process Action ---
        reward += self._handle_action(action)
        
        # --- Update Game World ---
        self._update_game_state()

        # --- Calculate Step Reward ---
        # (Rewards are handled inside _handle_action and _check_termination)
        
        # --- Check for Termination ---
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.game_over = terminated

        # MUST return exactly this 5-tuple
        truncated = self.steps >= self.MAX_STEPS
        terminated = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        step_reward = 0

        # --- Handle Selection ---
        if movement == 1: # Up
            self.selected_tool_idx = (self.selected_tool_idx - 1 + len(self.tools)) % len(self.tools)
            self.selected_snippet_idx = -1
        elif movement == 2: # Down
            self.selected_tool_idx = (self.selected_tool_idx + 1) % len(self.tools)
            self.selected_snippet_idx = -1
        elif movement == 3 and self.snippets: # Left
            self.selected_snippet_idx = (self.selected_snippet_idx - 1 + len(self.snippets)) % len(self.snippets)
        elif movement == 4 and self.snippets: # Right
            self.selected_snippet_idx = (self.selected_snippet_idx + 1) % len(self.snippets)

        # --- Handle Application ---
        apply_rhythm = 0
        if space_press: apply_rhythm = 1
        elif shift_press: apply_rhythm = 2
        
        if apply_rhythm > 0 and self.glitches:
            target_glitch = self.glitches[0] # Target the oldest glitch
            
            # --- Applying a Tool ---
            if self.selected_snippet_idx == -1:
                tool = self.tools[self.selected_tool_idx]
                
                # Correct application
                if tool['type'] == target_glitch['type'] and apply_rhythm == target_glitch['rhythm']:
                    # Sfx: success_chime.wav
                    self.glitch_level = max(0, self.glitch_level - 10)
                    step_reward += 1.0
                    self.score += 10
                    self._create_particles(target_glitch['pos'], tool['color'], 30, is_success=True)
                    self.action_feedback.append({'pos': target_glitch['pos'], 'color': self.COLOR_STABLE, 'life': 20})
                    self.glitches.pop(0)
                    step_reward += self._check_for_snippet_unlock(tool['type'])
                # Incorrect application
                else:
                    # Sfx: error_buzz.wav
                    self.glitch_level = min(self.MAX_GLITCH_LEVEL, self.glitch_level + 5)
                    step_reward -= 1.0
                    self.score -= 5
                    self.last_fix_chain.clear() # Break the chain on failure
                    self._create_particles(target_glitch['pos'], self.COLOR_UNSTABLE, 20, is_success=False)
                    self.action_feedback.append({'pos': target_glitch['pos'], 'color': self.COLOR_UNSTABLE, 'life': 20})

            # --- Applying a Snippet ---
            else:
                # Snippets are powerful and always succeed, consuming the snippet
                snippet = self.snippets.pop(self.selected_snippet_idx)
                pattern_to_fix = self.known_snippet_patterns[snippet['name']]
                
                glitches_to_remove = []
                for glitch in self.glitches:
                    if glitch['type'] in pattern_to_fix:
                        glitches_to_remove.append(glitch)
                
                for glitch in glitches_to_remove:
                    self.glitches.remove(glitch)
                    self.glitch_level = max(0, self.glitch_level - 10)
                    step_reward += 1.0
                    self.score += 20
                    self._create_particles(glitch['pos'], self.COLOR_SNIPPET, 40, is_success=True)
                
                self.selected_snippet_idx = -1 # Deselect snippets after use
        
        return step_reward
        
    def _update_game_state(self):
        # --- Update Glitches (animation) ---
        for glitch in self.glitches:
            glitch['life'] += 1

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['radius'] *= 0.95
            p['life'] -= 1
            
        # --- Update Action Feedback ---
        self.action_feedback = [f for f in self.action_feedback if f['life'] > 0]
        for f in self.action_feedback:
            f['life'] -= 1

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 200 == 0:
            self.glitch_spawn_rate = min(0.1, self.glitch_spawn_rate + 0.01)
            # Unlock new tool types
            if len(self.tools) < len(self.TOOL_COLORS):
                new_type = len(self.tools)
                self.tools.append({"name": f"Tracer v{new_type+1}", "type": new_type, "color": self.TOOL_COLORS[new_type]})

        # --- Spawn New Glitches ---
        if self.np_random.random() < self.glitch_spawn_rate and len(self.glitches) < 15:
            self._spawn_glitch()

    def _spawn_glitch(self):
        grid_x = self.np_random.integers(0, self.GRID_COLS)
        grid_y = self.np_random.integers(0, self.GRID_ROWS)
        pos = (
            grid_x * self.CELL_SIZE + self.CELL_SIZE // 2,
            grid_y * self.CELL_SIZE + self.CELL_SIZE // 2
        )
        
        # Ensure new glitch doesn't overlap existing
        while any(g['pos'] == pos for g in self.glitches):
            grid_x = self.np_random.integers(0, self.GRID_COLS)
            grid_y = self.np_random.integers(0, self.GRID_ROWS)
            pos = (
                grid_x * self.CELL_SIZE + self.CELL_SIZE // 2,
                grid_y * self.CELL_SIZE + self.CELL_SIZE // 2
            )

        glitch_type = self.np_random.integers(0, len(self.tools))
        glitch_rhythm = self.np_random.integers(1, 3) # 1 for space, 2 for shift
        
        self.glitches.append({
            "pos": pos,
            "type": glitch_type,
            "rhythm": glitch_rhythm,
            "life": 0,
            "size": self.CELL_SIZE * 0.6
        })

    def _check_for_snippet_unlock(self, fixed_glitch_type):
        self.last_fix_chain.append(fixed_glitch_type)
        chain_tuple = tuple(self.last_fix_chain)
        
        for name, pattern in self.known_snippet_patterns.items():
            if chain_tuple[-len(pattern):] == pattern:
                if not any(s['name'] == name for s in self.snippets):
                    # Sfx: unlock_snippet.wav
                    self.snippets.append({"name": name, "color": self.COLOR_SNIPPET})
                    self.last_fix_chain.clear()
                    self.score += 50
                    return 5.0 # Reward for creating a snippet
        return 0.0

    def _check_termination(self):
        if self.glitch_level >= self.MAX_GLITCH_LEVEL:
            return True, -100.0 # Failure
        if self.glitch_level <= 0 and not self.glitches:
            return True, 100.0 # Victory
        return False, 0.0

    def _get_observation(self):
        # --- Clear screen ---
        self.screen.fill(self.COLOR_BG)
        
        # --- Render all game elements ---
        self._render_background_grid()
        self._render_glitches()
        self._render_particles()
        self._render_action_feedback()
        
        # --- Render UI overlay ---
        self._render_ui()
        
        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.GAME_AREA_HEIGHT))
        for y in range(0, self.GAME_AREA_HEIGHT + 1, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_glitches(self):
        for i, glitch in enumerate(self.glitches):
            pos = (int(glitch['pos'][0]), int(glitch['pos'][1]))
            size = glitch['size'] * (1.0 + 0.1 * math.sin(glitch['life'] * 0.2))
            color = self.TOOL_COLORS[glitch['type']]

            # Different shape per type
            if glitch['type'] == 0: # Triangle
                points = [
                    (pos[0], pos[1] - size / 2),
                    (pos[0] - size / 2, pos[1] + size / 2),
                    (pos[0] + size / 2, pos[1] + size / 2)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            elif glitch['type'] == 1: # Square
                rect = pygame.Rect(pos[0] - size / 2, pos[1] - size / 2, size, size)
                pygame.draw.rect(self.screen, color, rect)
            else: # Circle
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(size / 2), color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(size / 2), color)

            # Draw rhythm indicator
            rhythm_char = "I" if glitch['rhythm'] == 1 else "II"
            rhythm_color = (255,255,255) if i == 0 else (150,150,150) # Highlight oldest
            self._draw_text(rhythm_char, (pos[0], pos[1] + size/2 + 5), 18, self.COLOR_TEXT, center=True)


    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p['color'])

    def _render_action_feedback(self):
        for f in self.action_feedback:
            alpha = int(255 * (f['life'] / 20))
            color = f['color'] + (alpha,)
            radius = int(self.CELL_SIZE * (1.0 - (f['life'] / 20)))
            
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(s, radius, radius, radius-1, color)
            self.screen.blit(s, (f['pos'][0] - radius, f['pos'][1] - radius))


    def _render_ui(self):
        ui_rect = pygame.Rect(0, self.GAME_AREA_HEIGHT, self.SCREEN_WIDTH, self.UI_HEIGHT)
        pygame.draw.rect(self.screen, (20, 20, 45), ui_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.GAME_AREA_HEIGHT), (self.SCREEN_WIDTH, self.GAME_AREA_HEIGHT), 2)
        
        # --- Glitch Level Meter ---
        self._draw_text("GLITCH LEVEL", (10, self.GAME_AREA_HEIGHT + 10), 20, self.COLOR_TEXT)
        bar_w = 200
        bar_h = 20
        bar_x = 10
        bar_y = self.GAME_AREA_HEIGHT + 35
        fill_w = int(bar_w * (self.glitch_level / self.MAX_GLITCH_LEVEL))
        
        # Interpolate color from green to red
        bar_color = (
            int(self.COLOR_STABLE[0] + (self.COLOR_UNSTABLE[0] - self.COLOR_STABLE[0]) * (self.glitch_level / self.MAX_GLITCH_LEVEL)),
            int(self.COLOR_STABLE[1] + (self.COLOR_UNSTABLE[1] - self.COLOR_STABLE[1]) * (self.glitch_level / self.MAX_GLITCH_LEVEL)),
            int(self.COLOR_STABLE[2] + (self.COLOR_UNSTABLE[2] - self.COLOR_STABLE[2]) * (self.glitch_level / self.MAX_GLITCH_LEVEL))
        )
        
        pygame.draw.rect(self.screen, (40,40,60), (bar_x, bar_y, bar_w, bar_h))
        if fill_w > 0: pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_w, bar_h), 1)

        # --- Score ---
        self._draw_text(f"SCORE: {self.score}", (self.SCREEN_WIDTH - 10, self.GAME_AREA_HEIGHT + 10), 24, self.COLOR_TEXT, align="topright")
        self._draw_text(f"STEPS: {self.steps}/{self.MAX_STEPS}", (self.SCREEN_WIDTH - 10, self.GAME_AREA_HEIGHT + 40), 20, self.COLOR_TEXT, align="topright")

        # --- Tool & Snippet Selector ---
        start_x = 240
        # Tools
        self._draw_text("TOOLS", (start_x, self.GAME_AREA_HEIGHT + 10), 20, self.COLOR_TEXT)
        for i, tool in enumerate(self.tools):
            rect = pygame.Rect(start_x + i * 50, self.GAME_AREA_HEIGHT + 35, 40, 30)
            pygame.draw.rect(self.screen, tool['color'], rect, border_radius=3)
            if i == self.selected_tool_idx and self.selected_snippet_idx == -1:
                pygame.draw.rect(self.screen, self.COLOR_STABLE, rect, 3, border_radius=3)
        
        # Snippets
        start_x_snippets = start_x + len(self.tools) * 50 + 20
        if self.snippets:
            self._draw_text("SNIPPETS", (start_x_snippets, self.GAME_AREA_HEIGHT + 10), 20, self.COLOR_TEXT)
            for i, snippet in enumerate(self.snippets):
                rect = pygame.Rect(start_x_snippets + i * 50, self.GAME_AREA_HEIGHT + 35, 40, 30)
                pygame.draw.rect(self.screen, snippet['color'], rect, border_radius=3)
                if i == self.selected_snippet_idx:
                    pygame.draw.rect(self.screen, self.COLOR_STABLE, rect, 3, border_radius=3)

    def _draw_text(self, text, pos, size, color, font_name=None, center=False, align="topleft"):
        font = pygame.font.Font(font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center: text_rect.center = pos
        elif align == "topleft": text_rect.topleft = pos
        elif align == "topright": text_rect.topright = pos
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, pos, color, count, is_success):
        for _ in range(count):
            if is_success:
                angle = self.np_random.random() * 2 * math.pi
                speed = self.np_random.random() * 2 + 1
                vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                radius = self.np_random.random() * 5 + 3
            else: # Failure particles are less energetic
                angle = self.np_random.random() * 2 * math.pi
                speed = self.np_random.random() * 1
                vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                radius = self.np_random.random() * 4 + 2

            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "radius": radius,
                "color": color,
                "life": self.np_random.integers(20, 40)
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "glitch_level": self.glitch_level,
            "active_glitches": len(self.glitches),
            "unlocked_tools": len(self.tools),
            "unlocked_snippets": len(self.snippets)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires the SDL_VIDEODRIVER to be set to a display-compatible value
    # e.g., by commenting out the os.environ line at the top of the file.
    
    # To run, comment out this line at the top of the file:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    try:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Glitch Stabilizer")
        clock = pygame.time.Clock()
        
        done = False
        total_reward = 0
        
        print("\n--- Manual Control ---")
        print(GameEnv.user_guide)
        print("R: Reset Environment | Q: Quit")

        while not done:
            # --- Human Input ---
            movement, space, shift = 0, 0, 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q: done = True
                    if event.key == pygame.K_r: 
                        obs, info = env.reset()
                        total_reward = 0
                    if event.key == pygame.K_UP: movement = 1
                    if event.key == pygame.K_DOWN: movement = 2
                    if event.key == pygame.K_LEFT: movement = 3
                    if event.key == pygame.K_RIGHT: movement = 4
                    if event.key == pygame.K_SPACE: space = 1
                    if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1

            action = [movement, space, shift]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward}")
                # Wait for user to quit or reset
                while True:
                    event = pygame.event.wait()
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        done = True
                        break
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        break
                if done: break # Exit outer loop if quitting

            # --- Rendering ---
            # The observation is already a rendered frame, so we just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Limit to 30 FPS for human playability
            
        env.close()

    except pygame.error as e:
        print("\nPygame display error. This is expected if you are running in a headless environment.")
        print("To play manually, comment out the 'os.environ' line at the top of the file and ensure you have a display.")
        
    # Test headless execution
    print("\nRunning a short headless test...")
    env = GameEnv()
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    env.close()
    print("Headless test completed successfully.")