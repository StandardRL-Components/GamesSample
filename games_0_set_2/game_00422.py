
# Generated: 2025-08-27T13:37:09.214634
# Source Brief: brief_00422.md
# Brief Index: 422

        
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
        "Controls: Arrow keys to move cursor. Space to place a crystal. Shift to cycle crystal type."
    )

    game_description = (
        "Strategically place limited crystals in an isometric cavern to refract light beams and illuminate all targets within the time limit."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.FPS * self.GAME_DURATION_SECONDS
        self.MAX_CRYSTALS = 20
        self.NUM_TARGETS = 5

        # --- Colors ---
        self.COLOR_BG = (15, 10, 30)
        self.COLOR_WALL = (40, 30, 70)
        self.COLOR_LIGHT = (255, 255, 100)
        self.COLOR_TARGET_OFF = (50, 50, 60)
        self.COLOR_TARGET_ON = (255, 255, 255)
        self.CRYSTAL_COLORS = [(255, 50, 50), (50, 100, 255), (50, 255, 100)] # Red, Blue, Green
        self.CRYSTAL_NAMES = ["Refractor (45°)", "Reflector (90°)", "Splitter (±30°)"]
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 255, 100)

        # --- Game Physics ---
        self.CURSOR_SPEED = 8
        self.CRYSTAL_RADIUS = 10
        self.TARGET_RADIUS = 15
        self.MAX_RAY_LENGTH = 1000 # Effectively infinite

        # --- Gym Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_big = pygame.font.Font(None, 64)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.cursor_pos = None
        self.crystals_remaining = 0
        self.placed_crystals = []
        self.selected_crystal_type = 0
        self.light_source = None
        self.targets = []
        self.light_paths = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_target_distances = {}
        self.cavern_bounds = pygame.Rect(50, 50, self.WIDTH - 100, self.HEIGHT - 100)

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        
        self.cursor_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.crystals_remaining = self.MAX_CRYSTALS
        self.placed_crystals = []
        self.selected_crystal_type = 0

        # Place light source on a random wall
        wall = self.np_random.integers(4)
        if wall == 0: # top
            self.light_source = {'pos': pygame.math.Vector2(self.np_random.uniform(self.cavern_bounds.left, self.cavern_bounds.right), self.cavern_bounds.top), 'dir': pygame.math.Vector2(0, 1)}
        elif wall == 1: # bottom
            self.light_source = {'pos': pygame.math.Vector2(self.np_random.uniform(self.cavern_bounds.left, self.cavern_bounds.right), self.cavern_bounds.bottom), 'dir': pygame.math.Vector2(0, -1)}
        elif wall == 2: # left
            self.light_source = {'pos': pygame.math.Vector2(self.cavern_bounds.left, self.np_random.uniform(self.cavern_bounds.top, self.cavern_bounds.bottom)), 'dir': pygame.math.Vector2(1, 0)}
        else: # right
            self.light_source = {'pos': pygame.math.Vector2(self.cavern_bounds.right, self.np_random.uniform(self.cavern_bounds.top, self.cavern_bounds.bottom)), 'dir': pygame.math.Vector2(-1, 0)}

        # Place targets randomly inside cavern
        self.targets = []
        while len(self.targets) < self.NUM_TARGETS:
            pos = pygame.math.Vector2(
                self.np_random.uniform(self.cavern_bounds.left + self.TARGET_RADIUS, self.cavern_bounds.right - self.TARGET_RADIUS),
                self.np_random.uniform(self.cavern_bounds.top + self.TARGET_RADIUS, self.cavern_bounds.bottom - self.TARGET_RADIUS)
            )
            # Ensure no overlap with other targets or source
            if all(pos.distance_to(t['pos']) > self.TARGET_RADIUS * 2.5 for t in self.targets) and pos.distance_to(self.light_source['pos']) > self.TARGET_RADIUS * 2:
                 self.targets.append({'pos': pos, 'is_lit': False})

        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._recalculate_light_and_targets()
        self._update_target_distances()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Input ---
        self._move_cursor(movement)
        
        # Cycle crystal type on shift press
        if shift_held and not self.prev_shift_held:
            self.selected_crystal_type = (self.selected_crystal_type + 1) % len(self.CRYSTAL_COLORS)
            # sfx: cycle_sound

        # Place crystal on space press
        placement_cost = 0
        newly_lit_reward = 0
        if space_held and not self.prev_space_held and self.crystals_remaining > 0:
            if self.cavern_bounds.collidepoint(self.cursor_pos) and not any(self.cursor_pos.distance_to(c['pos']) < self.CRYSTAL_RADIUS * 2 for c in self.placed_crystals):
                self.placed_crystals.append({'pos': pygame.math.Vector2(self.cursor_pos), 'type': self.selected_crystal_type})
                self.crystals_remaining -= 1
                placement_cost = -0.01 # Placement penalty
                # sfx: place_crystal
                newly_lit_reward = self._recalculate_light_and_targets()

        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        # --- Calculate Rewards ---
        # Distance reward
        new_distances = self._get_target_distances()
        distance_reward = 0
        for i, target in enumerate(self.targets):
            if not target['is_lit']:
                if i in new_distances and i in self.last_target_distances:
                    if new_distances[i] < self.last_target_distances[i]:
                        distance_reward += 0.1
        self.last_target_distances = new_distances
        
        # Combine rewards for this step
        reward += placement_cost
        reward += distance_reward
        reward += newly_lit_reward
        self.score += reward

        # --- Update State & Check Termination ---
        self.steps += 1
        terminated = False
        
        # Check for win
        if all(t['is_lit'] for t in self.targets):
            self.win_condition = True
            self.game_over = True
            terminated = True
            terminal_reward = 100
            self.score += terminal_reward
            reward += terminal_reward
        
        # Check for loss
        elif self.steps >= self.MAX_STEPS or (self.crystals_remaining == 0 and placement_cost == 0):
            self.game_over = True
            terminated = True
            terminal_reward = -100
            self.score += terminal_reward
            reward += terminal_reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _recalculate_light_and_targets(self):
        self.light_paths = self._calculate_light_paths()
        return self._update_target_illumination()

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

    def _calculate_light_paths(self):
        paths = []
        ray_queue = [(self.light_source['pos'], self.light_source['dir'])]
        
        processed_rays = 0
        while ray_queue and processed_rays < 50: # Safety break
            processed_rays += 1
            origin, direction = ray_queue.pop(0)
            
            hit = self._find_closest_intersection(origin, direction)
            
            if hit:
                end_point, hit_obj = hit
                paths.append((origin, end_point))
                
                if 'type' in hit_obj and hit_obj['type'] != 'wall': # It's a crystal
                    # sfx: refract_sound
                    crystal_type = hit_obj['type']
                    if crystal_type == 0: # Red: 45 degree turn
                        new_dir = direction.rotate(45)
                        ray_queue.append((end_point, new_dir))
                    elif crystal_type == 1: # Blue: 90 degree turn
                        new_dir = direction.rotate(90)
                        ray_queue.append((end_point, new_dir))
                    elif crystal_type == 2: # Green: Split +/- 30 degrees
                        dir1 = direction.rotate(30)
                        dir2 = direction.rotate(-30)
                        ray_queue.append((end_point, dir1))
                        ray_queue.append((end_point, dir2))
            else: # Hit nothing, goes to edge of render
                paths.append((origin, origin + direction * self.MAX_RAY_LENGTH))
        return paths

    def _find_closest_intersection(self, ray_origin, ray_dir):
        closest_hit = None
        min_dist = float('inf')

        # Check walls
        wall_points = [self.cavern_bounds.topleft, self.cavern_bounds.topright, self.cavern_bounds.bottomright, self.cavern_bounds.bottomleft]
        for i in range(4):
            p1 = pygame.math.Vector2(wall_points[i])
            p2 = pygame.math.Vector2(wall_points[(i + 1) % 4])
            
            v1 = ray_origin - p1
            v2 = p2 - p1
            v3 = pygame.math.Vector2(-ray_dir.y, ray_dir.x)
            
            dot = v2.dot(v3)
            if abs(dot) < 1e-6: continue

            t1 = v2.cross(v1) / dot
            t2 = v1.dot(v3) / dot

            if t1 >= 1e-4 and 0 <= t2 <= 1:
                if t1 < min_dist:
                    min_dist = t1
                    closest_hit = (ray_origin + t1 * ray_dir, {'type': 'wall'})
        
        # Check crystals
        for crystal in self.placed_crystals:
            oc = ray_origin - crystal['pos']
            b = 2 * oc.dot(ray_dir)
            c = oc.dot(oc) - self.CRYSTAL_RADIUS**2
            discriminant = b**2 - 4*c
            
            if discriminant >= 0:
                sqrt_d = math.sqrt(discriminant)
                t1 = (-b - sqrt_d) / 2
                
                if t1 > 1e-4:
                    if t1 < min_dist:
                        min_dist = t1
                        closest_hit = (ray_origin + t1 * ray_dir, crystal)
                        
        return closest_hit

    def _update_target_illumination(self):
        newly_lit_reward = 0
        for target in self.targets:
            was_lit = target['is_lit']
            if not was_lit:
                is_now_lit = False
                for p1, p2 in self.light_paths:
                    d = p2 - p1
                    if d.length_squared() == 0: continue
                    ft = ((target['pos'] - p1).dot(d)) / d.length_squared()
                    t = max(0, min(1, ft))
                    closest_point = p1 + t * d
                    if closest_point.distance_to(target['pos']) <= self.TARGET_RADIUS:
                        is_now_lit = True
                        break
                if is_now_lit:
                    target['is_lit'] = True
                    newly_lit_reward += 10 # Event-based reward
                    # sfx: target_lit
        return newly_lit_reward

    def _get_target_distances(self):
        distances = {}
        for i, target in enumerate(self.targets):
            if not target['is_lit']:
                min_dist = float('inf')
                for p1, p2 in self.light_paths:
                    d = p2 - p1
                    if d.length_squared() == 0: continue
                    ft = ((target['pos'] - p1).dot(d)) / d.length_squared()
                    t = max(0, min(1, ft))
                    closest_point = p1 + t * d
                    dist = closest_point.distance_to(target['pos'])
                    if dist < min_dist:
                        min_dist = dist
                if min_dist != float('inf'):
                    distances[i] = min_dist
        return distances

    def _update_target_distances(self):
        self.last_target_distances = self._get_target_distances()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw cavern walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, self.cavern_bounds, 3, border_radius=5)

        # Draw targets
        for target in self.targets:
            pos = (int(target['pos'].x), int(target['pos'].y))
            if target['is_lit']:
                self._draw_glow(self.screen, self.COLOR_TARGET_ON, pos, self.TARGET_RADIUS * 2, 100)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.TARGET_RADIUS, self.COLOR_TARGET_ON)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.TARGET_RADIUS, self.COLOR_TARGET_ON)
            else:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.TARGET_RADIUS, self.COLOR_TARGET_OFF)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.TARGET_RADIUS, self.COLOR_TARGET_OFF)

        # Draw light source
        pygame.gfxdraw.filled_circle(self.screen, int(self.light_source['pos'].x), int(self.light_source['pos'].y), 8, self.COLOR_LIGHT)

        # Draw light paths
        for p1, p2 in self.light_paths:
            self._draw_glow(self.screen, self.COLOR_LIGHT, p1, 10, 20, num_layers=1)
            pygame.draw.line(self.screen, self.COLOR_LIGHT, p1, p2, 3)
            pygame.draw.line(self.screen, (255,255,255), p1, p2, 1)

        # Draw placed crystals
        for crystal in self.placed_crystals:
            pos = (int(crystal['pos'].x), int(crystal['pos'].y))
            color = self.CRYSTAL_COLORS[crystal['type']]
            self._draw_glow(self.screen, color, pos, self.CRYSTAL_RADIUS * 2.5, 120)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CRYSTAL_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CRYSTAL_RADIUS, (255,255,255))
        
        # Draw cursor
        if not self.game_over:
            cursor_color = self.CRYSTAL_COLORS[self.selected_crystal_type]
            cursor_surf = pygame.Surface((self.CRYSTAL_RADIUS*2, self.CRYSTAL_RADIUS*2), pygame.SRCALPHA)
            pygame.draw.circle(cursor_surf, (*cursor_color, 100), (self.CRYSTAL_RADIUS, self.CRYSTAL_RADIUS), self.CRYSTAL_RADIUS)
            pygame.draw.circle(cursor_surf, (255,255,255,150), (self.CRYSTAL_RADIUS, self.CRYSTAL_RADIUS), self.CRYSTAL_RADIUS, 1)
            self.screen.blit(cursor_surf, self.cursor_pos - pygame.math.Vector2(self.CRYSTAL_RADIUS, self.CRYSTAL_RADIUS))

    def _render_ui(self):
        # Crystals remaining
        crystal_text = self.font_ui.render(f"Crystals: {self.crystals_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(crystal_text, (10, 10))
        
        # Selected Crystal Type
        type_text = self.font_ui.render(f"Selected: {self.CRYSTAL_NAMES[self.selected_crystal_type]}", True, self.CRYSTAL_COLORS[self.selected_crystal_type])
        self.screen.blit(type_text, (10, 30))

        # Timer
        time_left = max(0, self.GAME_DURATION_SECONDS - (self.steps / self.FPS))
        timer_text = self.font_ui.render(f"Time: {time_left:.1f}s", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Game Over message
        if self.game_over:
            msg = "YOU WIN!" if self.win_condition else "GAME OVER"
            color = (100, 255, 100) if self.win_condition else (255, 100, 100)
            end_text = self.font_big.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _draw_glow(self, surface, color, center, max_radius, alpha_start, num_layers=5):
        center_int = (int(center[0]), int(center[1]))
        for i in range(num_layers, 0, -1):
            radius = int(max_radius * (i / num_layers))
            if radius <= 0: continue
            alpha = int(alpha_start * (i / num_layers)**2)
            glow_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*color, alpha), (radius, radius), radius)
            surface.blit(glow_surf, (center_int[0] - radius, center_int[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "crystals_left": self.crystals_remaining}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    pygame.display.set_caption("Crystal Refractor")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        if not env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation from the environment to the display screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
        
        clock.tick(env.FPS)

        if terminated:
            print(f"Game Over. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for a moment before allowing a reset
            pygame.time.wait(2000)
            print("Press 'R' to play again or close the window.")
            
    env.close()