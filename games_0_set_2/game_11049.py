import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:24:31.189686
# Source Brief: brief_01049.md
# Brief Index: 1049
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
        "Terraform a shrinking world by solving anagrams. Collect letters to form words, "
        "stabilize the land, and expand your territory before it collapses."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to use the selected tool and "
        "shift to cycle between tools."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAYER_COLOR = (0, 255, 255)
    PLAYER_GLOW_COLOR = (150, 255, 255)
    STABLE_LAND_COLOR = (0, 255, 100)
    UNSTABLE_LAND_COLOR = (255, 50, 50)
    LAND_POLYGON_COLOR = (20, 30, 60)
    TOOL_EFFECT_COLOR = (255, 255, 0)
    BG_COLOR_TOP = (10, 5, 20)
    BG_COLOR_BOTTOM = (40, 10, 50)
    TEXT_COLOR = (220, 220, 240)
    HIGHLIGHT_COLOR = (255, 255, 100)
    
    PLAYER_SPEED = 8
    PLAYER_SIZE = 12
    MAX_STEPS = 5000
    
    WORD_LIST = ["AGENT", "SPACE", "WORLD", "TERRA", "FORM", "SHAPE", "SOLVE", "POINT"]

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
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        self.player_pos = np.array([0.0, 0.0])
        self.vertices = []
        self.particles = []
        self.anagram_letters = []
        self.unlocked_tools = []
        
        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # this is for debugging, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.world_center = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])
        self.player_pos = np.copy(self.world_center)
        
        self.shrink_rate = 0.1
        self.last_shrink_rate_increase_time = 0
        
        self._setup_landscape()
        self._setup_new_anagram()
        
        self.unlocked_tools = ["stabilize", "expand"]
        self.selected_tool_idx = 0
        
        self.last_space_state = 0
        self.last_shift_state = 0
        
        self.particles = []
        self.survival_bonus_milestone = 1800 # 60 seconds at 30fps

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Base cost of existence

        reward += self._handle_input(action)
        self._update_game_state()
        reward += self._check_letter_collection()

        # Survival bonus
        if self.steps >= self.survival_bonus_milestone:
            reward += 50
            self.survival_bonus_milestone += 1800
        
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            self.game_over = True
            if self._calculate_land_area() < 100:
                reward -= 100 # Penalty for losing

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Movement
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] -= 1
        elif movement == 2: move_vec[1] += 1
        elif movement == 3: move_vec[0] -= 1
        elif movement == 4: move_vec[0] += 1
        
        if np.linalg.norm(move_vec) > 0:
            self.player_pos += move_vec * self.PLAYER_SPEED

        # Cycle tool on SHIFT press
        if shift_held and not self.last_shift_state:
            self.selected_tool_idx = (self.selected_tool_idx + 1) % len(self.unlocked_tools)
            # sfx: UI_CycleTool
        self.last_shift_state = shift_held

        # Apply tool on SPACE press
        if space_held and not self.last_space_state:
            reward += self._apply_tool()
            # sfx: Tool_Use
        self.last_space_state = space_held
        
        return reward

    def _update_game_state(self):
        # Update landscape shrinking
        if self.steps - self.last_shrink_rate_increase_time > 1800: # 60 seconds
            self.shrink_rate += 0.05
            self.last_shrink_rate_increase_time = self.steps

        for v in self.vertices:
            if not v['stable']:
                v['radius'] = max(0, v['radius'] - self.shrink_rate / 30.0)
            v['pos'] = self.world_center + np.array([math.cos(v['angle']), math.sin(v['angle'])]) * v['radius']

        # Clamp player position to a generous boundary around the world center
        max_dist = max(v['radius'] for v in self.vertices) if self.vertices else 1
        dist_from_center = np.linalg.norm(self.player_pos - self.world_center)
        if dist_from_center > max_dist + self.PLAYER_SIZE:
             self.player_pos = self.world_center + (self.player_pos - self.world_center) / dist_from_center * (max_dist + self.PLAYER_SIZE)

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.95

    def _apply_tool(self):
        tool_name = self.unlocked_tools[self.selected_tool_idx]
        
        # Find closest vertex to player
        if not self.vertices: return 0
        
        dists = [np.linalg.norm(self.player_pos - v['pos']) for v in self.vertices]
        closest_idx = np.argmin(dists)
        closest_v = self.vertices[closest_idx]

        if dists[closest_idx] > 50: # Tool range
            return 0
        
        reward = 0
        if tool_name == "stabilize":
            if not closest_v['stable']:
                closest_v['stable'] = True
                self._create_particles(closest_v['pos'], self.STABLE_LAND_COLOR, 20)
                reward = 0.1
                # sfx: Stabilize_Success
        elif tool_name == "expand":
            closest_v['radius'] += 20
            closest_v['stable'] = False # Expanding makes it unstable again
            self._create_particles(closest_v['pos'], self.TOOL_EFFECT_COLOR, 30)
            reward = 0.05
            # sfx: Expand_Success
        return reward

    def _check_letter_collection(self):
        if not self.anagram_letters or self.anagram_progress >= len(self.current_anagram_word):
            return 0
        
        target_letter = self.anagram_letters[self.anagram_progress]
        dist = np.linalg.norm(self.player_pos - target_letter['pos'])
        
        if dist < self.PLAYER_SIZE + 10:
            self.anagram_progress += 1
            self._create_particles(target_letter['pos'], self.HIGHLIGHT_COLOR, 15)
            # sfx: Letter_Collect
            
            if self.anagram_progress >= len(self.current_anagram_word):
                # Anagram solved!
                reward = 5.0
                self.shrink_rate = max(0.05, self.shrink_rate * 0.8) # Slow down shrink rate
                
                # Unlock new tool if any
                if len(self.unlocked_tools) < 2 and "expand" not in self.unlocked_tools:
                    self.unlocked_tools.append("expand")
                    reward += 1.0

                self._setup_new_anagram()
                # sfx: Anagram_Solved
                return reward
        return 0

    def _setup_landscape(self):
        self.vertices = []
        num_vertices = 8
        initial_radius = 180
        for i in range(num_vertices):
            angle = (2 * math.pi * i / num_vertices) + (math.pi / num_vertices)
            pos = self.world_center + np.array([math.cos(angle), math.sin(angle)]) * initial_radius
            self.vertices.append({
                'angle': angle,
                'radius': initial_radius,
                'pos': pos,
                'stable': False
            })
        self.initial_land_area = self._calculate_land_area()

    def _setup_new_anagram(self):
        self.current_anagram_word = self.np_random.choice(self.WORD_LIST)
        scrambled = list(self.current_anagram_word)
        self.np_random.shuffle(scrambled)
        self.scrambled_word = "".join(scrambled)
        self.anagram_progress = 0
        self.anagram_letters = []

        if not self.vertices: return

        available_vertices = list(range(len(self.vertices)))
        self.np_random.shuffle(available_vertices)
        
        for i, char in enumerate(self.current_anagram_word):
            if not available_vertices: break
            
            # Place letters between two vertices
            v_idx1 = available_vertices.pop()
            v_idx2 = (v_idx1 + 1) % len(self.vertices)
            
            v1 = self.vertices[v_idx1]
            v2 = self.vertices[v_idx2]
            
            # Place letter at midpoint of the edge, slightly inwards
            mid_point_radius = (v1['radius'] + v2['radius']) / 2 * 0.8
            mid_point_angle = v1['angle'] + (v2['angle'] - v1['angle']) / 2
            
            pos = self.world_center + np.array([math.cos(mid_point_angle), math.sin(mid_point_angle)]) * mid_point_radius
            
            self.anagram_letters.append({
                'char': char,
                'pos': pos
            })

    def _check_termination(self):
        if self._calculate_land_area() < 100: # Effectively zero area
            return True
        return False

    def _get_observation(self):
        self._render_to_surface()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "land_area_percent": (self._calculate_land_area() / self.initial_land_area * 100) if self.initial_land_area > 0 else 0
        }

    def _render_to_surface(self):
        # --- Background ---
        self._draw_gradient_background()
        
        # --- Dynamic Zoom and Centering ---
        scale, offset = self._get_transform()
        
        # --- Game Elements ---
        self._draw_landscape(scale, offset)
        self._draw_particles(scale, offset)
        self._draw_letters(scale, offset)
        self._draw_player(scale, offset)
        
        # --- UI Overlay ---
        self._render_ui()

    def _get_transform(self):
        if not self.vertices:
            return 1.0, np.array([0, 0])
        
        min_coords = np.min([v['pos'] for v in self.vertices], axis=0)
        max_coords = np.max([v['pos'] for v in self.vertices], axis=0)
        
        bbox_size = max_coords - min_coords
        if bbox_size[0] < 1 or bbox_size[1] < 1:
            return 1.0, self.world_center - self.player_pos

        margin = 1.4
        scale_x = self.SCREEN_WIDTH / (bbox_size[0] * margin)
        scale_y = self.SCREEN_HEIGHT / (bbox_size[1] * margin)
        scale = min(scale_x, scale_y)
        
        bbox_center = min_coords + bbox_size / 2.0
        offset = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0]) - bbox_center * scale
        
        return scale, offset

    def _transform_point(self, pos, scale, offset):
        return pos * scale + offset

    def _draw_gradient_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.BG_COLOR_TOP[0] * (1 - interp) + self.BG_COLOR_BOTTOM[0] * interp),
                int(self.BG_COLOR_TOP[1] * (1 - interp) + self.BG_COLOR_BOTTOM[1] * interp),
                int(self.BG_COLOR_TOP[2] * (1 - interp) + self.BG_COLOR_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_landscape(self, scale, offset):
        if len(self.vertices) < 3: return
        
        center_screen = self._transform_point(self.world_center, scale, offset)
        
        # Draw connections
        for v in self.vertices:
            v_screen = self._transform_point(v['pos'], scale, offset)
            color = self.STABLE_LAND_COLOR if v['stable'] else self.UNSTABLE_LAND_COLOR
            
            # Pulse unstable lines
            if not v['stable']:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                color = (
                    int(color[0] * (0.6 + 0.4 * pulse)),
                    int(color[1] * (0.6 + 0.4 * pulse)),
                    int(color[2] * (0.6 + 0.4 * pulse))
                )
            
            pygame.draw.aaline(self.screen, color, center_screen, v_screen, 1)

        # Draw main polygon
        poly_points = [self._transform_point(v['pos'], scale, offset) for v in self.vertices]
        pygame.gfxdraw.filled_polygon(self.screen, poly_points, self.LAND_POLYGON_COLOR)
        pygame.gfxdraw.aapolygon(self.screen, poly_points, self.STABLE_LAND_COLOR)

    def _draw_player(self, scale, offset):
        player_screen_pos = self._transform_point(self.player_pos, scale, offset)
        size = self.PLAYER_SIZE
        
        # Create a triangle pointing up
        points = [
            (player_screen_pos[0], player_screen_pos[1] - size * 1.2),
            (player_screen_pos[0] - size, player_screen_pos[1] + size * 0.6),
            (player_screen_pos[0] + size, player_screen_pos[1] + size * 0.6)
        ]
        
        # Glow effect
        for i in range(5, 0, -1):
            glow_size = size + i * 2
            glow_alpha = 50 - i * 8
            glow_points = [
                (player_screen_pos[0], player_screen_pos[1] - glow_size * 1.2),
                (player_screen_pos[0] - glow_size, player_screen_pos[1] + glow_size * 0.6),
                (player_screen_pos[0] + glow_size, player_screen_pos[1] + glow_size * 0.6)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, glow_points, (*self.PLAYER_GLOW_COLOR, glow_alpha))
        
        pygame.gfxdraw.filled_polygon(self.screen, points, self.PLAYER_COLOR)
        pygame.gfxdraw.aapolygon(self.screen, points, self.PLAYER_COLOR)

    def _draw_letters(self, scale, offset):
        for i, letter in enumerate(self.anagram_letters):
            is_target = (i == self.anagram_progress)
            
            pos = self._transform_point(letter['pos'], scale, offset)
            color = self.HIGHLIGHT_COLOR if is_target else self.TEXT_COLOR
            
            font_surface = self.font_medium.render(letter['char'], True, color)
            
            if is_target:
                pulse = (math.sin(self.steps * 0.3) + 1) / 2 * 0.2 + 1.0 # 1.0 to 1.2
                font_surface = pygame.transform.rotozoom(font_surface, 0, pulse)
            
            rect = font_surface.get_rect(center=(int(pos[0]), int(pos[1])))
            self.screen.blit(font_surface, rect)

    def _draw_particles(self, scale, offset):
        for p in self.particles:
            pos = self._transform_point(p['pos'], scale, offset)
            alpha = int(255 * (p['life'] / p['max_life']))
            if alpha > 0:
                color = (*p['color'], alpha)
                radius = int(max(1, p['radius'] * scale))
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, color)

    def _render_ui(self):
        # --- Score and Steps ---
        score_text = self.font_small.render(f"SCORE: {self.score:.2f}", True, self.TEXT_COLOR)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.TEXT_COLOR)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        # --- Anagram Display ---
        y_pos = 40
        x_pos = self.SCREEN_WIDTH / 2
        for i, char in enumerate(self.current_anagram_word):
            color = self.STABLE_LAND_COLOR if i < self.anagram_progress else self.TEXT_COLOR
            char_surf = self.font_large.render(char, True, color)
            char_rect = char_surf.get_rect(center=(x_pos, y_pos))
            self.screen.blit(char_surf, char_rect)
            x_pos += 25
        
        # --- Tool Display ---
        tool_y = self.SCREEN_HEIGHT - 30
        for i, tool_name in enumerate(self.unlocked_tools):
            is_selected = (i == self.selected_tool_idx)
            color = self.HIGHLIGHT_COLOR if is_selected else self.TEXT_COLOR
            text = self.font_medium.render(tool_name.upper(), True, color)
            rect = text.get_rect(center=(self.SCREEN_WIDTH / 2 - 60 + 120 * i, tool_y))
            
            if is_selected:
                pygame.draw.rect(self.screen, (*self.HIGHLIGHT_COLOR, 50), rect.inflate(20, 10), border_radius=5)

            self.screen.blit(text, rect)

        # --- World Integrity Bar ---
        area_percent = (self._calculate_land_area() / self.initial_land_area) if self.initial_land_area > 0 else 0
        bar_width = 200
        bar_height = 10
        bar_x = self.SCREEN_WIDTH / 2 - bar_width / 2
        bar_y = 10
        
        pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        fill_color = self.STABLE_LAND_COLOR if area_percent > 0.3 else self.UNSTABLE_LAND_COLOR
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, bar_width * area_percent, bar_height), border_radius=3)


    def _calculate_land_area(self):
        if len(self.vertices) < 3:
            return 0
        points = [v['pos'] for v in self.vertices]
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': np.copy(pos),
                'vel': np.array([math.cos(angle), math.sin(angle)]) * speed,
                'life': life,
                'max_life': life,
                'radius': self.np_random.uniform(2, 5),
                'color': color
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual play and debugging.
    # It will not be executed by the test suite.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible driver for manual play
    
    env = GameEnv()
    obs, info = env.reset()
    terminated, truncated = False, False
    
    display = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Anagram Terraformer")

    # To use MultiDiscrete with keyboard, we simulate it
    action = [0, 0, 0] # [movement, space, shift]

    while not terminated and not truncated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        action[0] = 0 # No movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)

        # For human play, we need to render to the screen
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        display.blit(render_surface, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Lock to 30 FPS

    env.close()