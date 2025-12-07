import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:43:54.298760
# Source Brief: brief_01280.md
# Brief Index: 1280
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A puzzle game where you use portals and polarity-flipping tools to make all cogs spin clockwise."
    )
    user_guide = (
        "Controls: ←→ to select a portal, ↑↓ to select a cog. Use Shift to cycle tools and Space to activate the selected tool at the selected portal."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Fonts and Colors
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_tool = pygame.font.SysFont("monospace", 22, bold=True)
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_BG_GEAR = (30, 35, 40)
        self.COLOR_COG = (180, 140, 80)
        self.COLOR_COG_CENTER = (150, 110, 60)
        self.COLOR_CW = (100, 150, 255) # Blue
        self.COLOR_CCW = (255, 100, 100) # Red
        self.COLOR_PORTAL = (150, 255, 150) # Green
        self.COLOR_SELECT = (255, 220, 50) # Gold
        self.COLOR_TEXT = (220, 220, 220)

        # Game Constants
        self.MAX_TURNS_PER_PUZZLE = 50
        self.MAX_EPISODE_STEPS = 5000
        self.COG_RADIUS_RANGE = (25, 40)
        self.PORTAL_RADIUS = 15

        # Persistent State (across resets)
        self.puzzles_solved = 0
        self._define_tools()
        self.unlocked_tools = [self.all_tools[0]]

        # Per-Puzzle State (initialized in reset)
        self.steps = 0
        self.turn_counter = 0
        self.score = 0
        self.game_over = False
        self.cogs = []
        self.portals = []
        self.particles = []
        self.background_gears = []
        self.selected_cog_idx = 0
        self.selected_portal_idx = 0
        self.selected_tool_idx = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()
        # self.validate_implementation() # Removed for production

    def _define_tools(self):
        self.all_tools = [
            {'name': 'Polarity Flipper S', 'radius': 80, 'color': (255, 100, 100)},
            {'name': 'Polarity Flipper M', 'radius': 120, 'color': (255, 160, 100)},
            {'name': 'Polarity Flipper L', 'radius': 180, 'color': (255, 220, 100)},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.turn_counter = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        
        self._update_progression()
        self._generate_puzzle()

        self.selected_cog_idx = 0 if self.cogs else -1
        self.selected_portal_idx = 0 if self.portals else -1
        self.selected_tool_idx = 0
        
        return self._get_observation(), self._get_info()

    def _update_progression(self):
        # Unlock new tools every 5 puzzles
        new_tool_idx = self.puzzles_solved // 5
        if new_tool_idx < len(self.all_tools) and self.all_tools[new_tool_idx] not in self.unlocked_tools:
            self.unlocked_tools.append(self.all_tools[new_tool_idx])

    def _generate_puzzle(self):
        self.cogs = []
        self.portals = []

        num_cogs = min(10, 3 + self.puzzles_solved // 3)
        num_portals = min(5, 2 + self.puzzles_solved // 6)
        
        # Generate non-overlapping cogs
        for _ in range(num_cogs):
            attempts = 0
            while attempts < 100:
                radius = self.np_random.integers(self.COG_RADIUS_RANGE[0], self.COG_RADIUS_RANGE[1])
                pos = (self.np_random.integers(radius, self.screen_width - radius), 
                       self.np_random.integers(radius, self.screen_height - radius - 50))
                
                # Check for overlap with existing cogs
                is_overlapping = False
                for other_cog in self.cogs:
                    dist = math.hypot(pos[0] - other_cog['pos'][0], pos[1] - other_cog['pos'][1])
                    if dist < radius + other_cog['radius'] + 20: # +20 for spacing
                        is_overlapping = True
                        break
                
                if not is_overlapping:
                    self.cogs.append({
                        'pos': pos, 
                        'radius': radius, 
                        'is_clockwise': self.np_random.choice([True, False]),
                        'rotation': self.np_random.uniform(0, 360)
                    })
                    break
                attempts += 1
        
        # Generate non-overlapping portals
        for _ in range(num_portals):
            attempts = 0
            while attempts < 100:
                pos = (self.np_random.integers(self.PORTAL_RADIUS, self.screen_width - self.PORTAL_RADIUS), 
                       self.np_random.integers(self.PORTAL_RADIUS, self.screen_height - self.PORTAL_RADIUS - 50))
                
                is_overlapping = any(math.hypot(pos[0] - p['pos'][0], pos[1] - p['pos'][1]) < self.PORTAL_RADIUS * 3 for p in self.portals)
                
                if not is_overlapping:
                    self.portals.append({'pos': pos})
                    break
                attempts += 1
        
        # Generate background gears for atmosphere
        self.background_gears = []
        for _ in range(5):
            self.background_gears.append({
                'pos': (self.np_random.integers(0, self.screen_width), self.np_random.integers(0, self.screen_height)),
                'radius': self.np_random.integers(80, 200),
                'speed': self.np_random.uniform(0.01, 0.05) * self.np_random.choice([-1, 1]),
                'rotation': self.np_random.uniform(0, 360)
            })

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False
        truncated = False

        # --- Handle player input (continuous) ---
        if movement != 0:
            if movement == 1 and self.cogs: self.selected_cog_idx = (self.selected_cog_idx - 1) % len(self.cogs)
            elif movement == 2 and self.cogs: self.selected_cog_idx = (self.selected_cog_idx + 1) % len(self.cogs)
            elif movement == 3 and self.portals: self.selected_portal_idx = (self.selected_portal_idx - 1) % len(self.portals)
            elif movement == 4 and self.portals: self.selected_portal_idx = (self.selected_portal_idx + 1) % len(self.portals)

        if shift_held and not self.last_shift_held and self.unlocked_tools:
            self.selected_tool_idx = (self.selected_tool_idx + 1) % len(self.unlocked_tools)

        # --- Handle main action (turn-based on space press) ---
        if space_held and not self.last_space_held and self.portals and self.unlocked_tools:
            self.turn_counter += 1
            reward_change, is_puzzle_over = self._apply_tool_and_get_reward()
            reward += reward_change
            self.score += reward_change
            if is_puzzle_over:
                terminated = True

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        self.steps += 1
        
        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
        
        # If the puzzle is over, we reset it for the next one
        if terminated:
            self.reset()
            terminated = False # The episode continues, but the puzzle is reset

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _apply_tool_and_get_reward(self):
        tool = self.unlocked_tools[self.selected_tool_idx]
        portal = self.portals[self.selected_portal_idx]
        portal_pos = portal['pos']
        
        cogs_affected = 0
        for cog in self.cogs:
            dist = math.hypot(cog['pos'][0] - portal_pos[0], cog['pos'][1] - portal_pos[1])
            if dist <= tool['radius']:
                cog['is_clockwise'] = not cog['is_clockwise']
                cogs_affected += 1
        
        if cogs_affected > 0:
            self._create_particles(portal_pos, tool['color'], 30)

        # --- Calculate reward and check for termination ---
        reward = 0
        num_cw = sum(1 for c in self.cogs if c['is_clockwise'])
        
        # Continuous reward for correct spins
        reward += num_cw

        # Check for win condition
        if num_cw == len(self.cogs):
            reward += 100  # Win bonus
            self.puzzles_solved += 1
            # Check for tool unlock bonus
            if self.puzzles_solved > 0 and self.puzzles_solved % 5 == 0 and (self.puzzles_solved // 5) < len(self.all_tools):
                reward += 50
            return reward, True # Puzzle is over

        # Check for loss condition
        if self.turn_counter >= self.MAX_TURNS_PER_PUZZLE:
            reward -= 100 # Loss penalty
            return reward, True # Puzzle is over
            
        return float(reward), False

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
            "turn": self.turn_counter,
            "puzzles_solved": self.puzzles_solved
        }
        
    def _render_game(self):
        # Update and draw background elements
        for gear in self.background_gears:
            gear['rotation'] = (gear['rotation'] + gear['speed']) % 360
            self._draw_cog(self.screen, gear['pos'][0], gear['pos'][1], gear['radius'], 20, gear['rotation'], self.COLOR_BG_GEAR, self.COLOR_BG_GEAR, is_bg=True)

        # Draw portals
        if self.portals:
            for i, portal in enumerate(self.portals):
                self._draw_portal(self.screen, portal['pos'], i == self.selected_portal_idx)
            
            # Draw selection effect for portal
            if self.selected_portal_idx != -1 and self.unlocked_tools:
                selected_portal_pos = self.portals[self.selected_portal_idx]['pos']
                tool_radius = self.unlocked_tools[self.selected_tool_idx]['radius']
                self._draw_selection_aoe(self.screen, selected_portal_pos, tool_radius)

        # Draw cogs
        if self.cogs:
            for i, cog in enumerate(self.cogs):
                spin_speed = 2.0
                if cog['is_clockwise']:
                    cog['rotation'] = (cog['rotation'] + spin_speed) % 360
                    spin_color = self.COLOR_CW
                else:
                    cog['rotation'] = (cog['rotation'] - spin_speed) % 360
                    spin_color = self.COLOR_CCW
                
                is_selected = (i == self.selected_cog_idx)
                self._draw_cog(self.screen, cog['pos'][0], cog['pos'][1], cog['radius'], 12, cog['rotation'], self.COLOR_COG, spin_color, is_selected)
        
        self._update_and_draw_particles()

    def _render_ui(self):
        # Draw a semi-transparent bar at the bottom
        ui_bar = pygame.Surface((self.screen_width, 50), pygame.SRCALPHA)
        ui_bar.fill((10, 15, 20, 200))
        self.screen.blit(ui_bar, (0, self.screen_height - 50))
        
        # Display info
        turn_text = self.font_ui.render(f"Turn: {self.turn_counter}/{self.MAX_TURNS_PER_PUZZLE}", True, self.COLOR_TEXT)
        score_text = self.font_ui.render(f"Score: {self.score:.0f}", True, self.COLOR_TEXT)
        puzzles_text = self.font_ui.render(f"Solved: {self.puzzles_solved}", True, self.COLOR_TEXT)
        
        self.screen.blit(turn_text, (15, self.screen_height - 38))
        self.screen.blit(score_text, (200, self.screen_height - 38))
        self.screen.blit(puzzles_text, (380, self.screen_height - 38))
        
        # Display current tool
        if self.unlocked_tools:
            tool = self.unlocked_tools[self.selected_tool_idx]
            tool_text = self.font_tool.render(f"Tool: {tool['name']}", True, tool['color'])
            text_rect = tool_text.get_rect(center=(self.screen_width / 2, 25))
            self.screen.blit(tool_text, text_rect)

    def _draw_cog(self, surface, x, y, radius, num_teeth, angle, color, spin_color, is_selected=False, is_bg=False):
        if is_selected:
            self._draw_glow(surface, (x, y), radius + 10, 5, self.COLOR_SELECT)

        points = []
        tooth_height = radius / 8
        for i in range(num_teeth * 2):
            r = radius if i % 2 == 0 else radius + tooth_height
            current_angle = math.radians(angle + i * (360 / (num_teeth * 2)))
            px = int(x + r * math.cos(current_angle))
            py = int(y + r * math.sin(current_angle))
            points.append((px, py))
        
        if points:
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)

        # Center circle
        pygame.gfxdraw.filled_circle(surface, int(x), int(y), int(radius * 0.5), self.COLOR_COG_CENTER if not is_bg else self.COLOR_BG_GEAR)
        pygame.gfxdraw.aacircle(surface, int(x), int(y), int(radius * 0.5), self.COLOR_COG_CENTER if not is_bg else self.COLOR_BG_GEAR)

        # Spin indicator arrow
        if not is_bg:
            arrow_angle = math.radians(angle + 90 if spin_color == self.COLOR_CW else angle - 90)
            start_pos = (int(x + radius * 0.7 * math.cos(arrow_angle)), int(y + radius * 0.7 * math.sin(arrow_angle)))
            
            # Simple triangle for arrow
            p1 = start_pos
            p2 = (int(p1[0] - 8 * math.cos(arrow_angle - 0.5)), int(p1[1] - 8 * math.sin(arrow_angle - 0.5)))
            p3 = (int(p1[0] - 8 * math.cos(arrow_angle + 0.5)), int(p1[1] - 8 * math.sin(arrow_angle + 0.5)))
            pygame.gfxdraw.filled_trigon(surface, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], spin_color)

    def _draw_portal(self, surface, pos, is_selected):
        if is_selected:
            self._draw_glow(surface, pos, self.PORTAL_RADIUS + 10, 5, self.COLOR_SELECT)

        # Shimmering effect
        for i in range(3):
            phase = (self.steps * 0.1 + i * math.pi / 2)
            radius = self.PORTAL_RADIUS * (1 + 0.1 * math.sin(phase))
            alpha = 100 + 80 * math.cos(phase)
            color = (*self.COLOR_PORTAL, alpha)
            
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, int(radius), int(radius), int(radius), color)
            surface.blit(temp_surf, (int(pos[0] - radius), int(pos[1] - radius)))

    def _draw_selection_aoe(self, surface, pos, radius):
        # Draw a transparent circle indicating the tool's area of effect
        s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_SELECT, 30), pos, radius)
        pygame.draw.circle(s, (*self.COLOR_SELECT, 60), pos, radius, width=2)
        surface.blit(s, (0,0))
        
    def _draw_glow(self, surface, pos, max_radius, steps, color):
        for i in range(steps):
            radius = max_radius * (i / steps)
            alpha = 50 * (1 - (i / steps))**2
            s = pygame.Surface((max_radius * 2, max_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (max_radius, max_radius), radius)
            surface.blit(s, (pos[0] - max_radius, pos[1] - max_radius))

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifetime = self.np_random.integers(20, 40)
            self.particles.append({'pos': list(pos), 'vel': velocity, 'life': lifetime, 'max_life': lifetime, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / p['max_life']))
                color = (*p['color'], alpha)
                size = int(5 * (p['life'] / p['max_life']))
                if size > 0:
                    rect = pygame.Rect(int(p['pos'][0] - size/2), int(p['pos'][1] - size/2), size, size)
                    pygame.draw.rect(self.screen, color, rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Example usage to visualize the environment
    # This part requires a display. It will not run in a headless environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Cogworks Portal Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Map keyboard keys to actions for human play
        keys = pygame.key.get_pressed()
        movement = 0
        
        # We check for keydown events to avoid rapid selection
        action_movement = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action_movement = 1
                elif event.key == pygame.K_DOWN: action_movement = 2
                elif event.key == pygame.K_LEFT: action_movement = 3
                elif event.key == pygame.K_RIGHT: action_movement = 4
                elif event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [action_movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode Over! Final Score: {info['score']}. Puzzles Solved: {info['puzzles_solved']}")
            obs, info = env.reset()

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()