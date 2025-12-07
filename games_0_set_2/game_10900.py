import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:08:07.620996
# Source Brief: brief_00900.md
# Brief Index: 900
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Cyberpunk Rhythm-Stealth Roguelike Gymnasium Environment.

    The player must navigate a procedurally generated corporate HQ to reach a
    data server. They can deploy clones to distract guards and use portals
    for teleportation. Detection by a guard results in immediate failure.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Navigate a corporate HQ to reach a data server, using clones and portals to avoid guards."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move. Press space to deploy a clone and shift to place a portal."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 20
    GRID_WIDTH, GRID_HEIGHT = SCREEN_WIDTH // GRID_SIZE, SCREEN_HEIGHT // GRID_SIZE

    # Colors (Cyberpunk Neon)
    COLOR_BG = (10, 10, 26) # Dark blue-purple
    COLOR_GRID = (30, 30, 50)
    COLOR_WALL = (150, 150, 170)
    COLOR_WALL_GLOW = (180, 180, 200)

    COLOR_PLAYER = (0, 255, 255) # Cyan
    COLOR_PLAYER_GLOW = (0, 255, 255, 50)

    COLOR_OBJECTIVE = (255, 255, 0) # Yellow
    COLOR_OBJECTIVE_GLOW = (255, 255, 0, 80)

    COLOR_ENEMY = (255, 50, 100) # Hot Pink/Red
    COLOR_ENEMY_GLOW = (255, 50, 100, 50)
    COLOR_DETECTION_CONE = (255, 0, 0, 40)
    COLOR_DISTRACTED_CONE = (255, 255, 0, 40)

    COLOR_CLONE = (255, 165, 0) # Orange
    COLOR_CLONE_GLOW = (255, 165, 0, 80)

    COLOR_PORTAL_1 = (0, 100, 255) # Blue
    COLOR_PORTAL_2 = (200, 0, 255) # Purple

    COLOR_UI_TEXT = (220, 220, 220)

    # Game Parameters
    MAX_STEPS = 1000
    CLONE_LIFESPAN = 15 # steps
    ENEMY_SIGHT_RANGE = 8 # grid units
    ENEMY_CONE_ANGLE = math.pi / 2.5 # 72 degrees

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

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
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self.player_pos = None
        self.objective_pos = None
        self.walls = None
        self.enemies = None
        self.clones = None
        self.portals = None
        self.particles = None

        self.prev_space_held = False
        self.prev_shift_held = False
        
        # This will be called once to ensure the env is set up correctly.
        # self.reset() is called by the test harness, no need to call it here.


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self._generate_level()

        self.clones = []
        self.portals = []
        self.particles = []

        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input and State Updates ---
        old_pos = self.player_pos
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        # Player actions
        reward += self._handle_input(movement, space_pressed, shift_pressed)
        
        # Update game objects
        distracted_an_enemy = self._update_clones()
        self._update_enemies()
        self._update_particles()
        
        # Check for detection
        detected, in_cone = self._check_player_detection()
        if detected:
            self.game_over = True
            # Sound: Detection alarm
            reward -= 100.0
        elif in_cone:
            reward -= 0.5

        # Check for win condition
        if self.player_pos == self.objective_pos:
            self.game_over = True
            self.game_won = True
            # Sound: Success chime
            reward += 100.0

        # Calculate distance-based reward
        dist_old = self._get_manhattan_distance(old_pos, self.objective_pos)
        dist_new = self._get_manhattan_distance(self.player_pos, self.objective_pos)
        reward += (dist_old - dist_new) * 0.1

        # Add event-based rewards
        if distracted_an_enemy:
            reward += 5.0
        
        self.score += reward
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # Check for episode termination
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS
        if truncated and not self.game_over:
            # Penalize for running out of time
            self.score -= 10
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

    # --- Game Logic Methods ---

    def _generate_level(self):
        while True:
            self.walls = set()
            # Add borders
            for x in range(self.GRID_WIDTH):
                self.walls.add((x, 0))
                self.walls.add((x, self.GRID_HEIGHT - 1))
            for y in range(self.GRID_HEIGHT):
                self.walls.add((0, y))
                self.walls.add((self.GRID_WIDTH - 1, y))

            # Generate some random wall obstacles
            num_obstacles = self.np_random.integers(5, 10)
            for _ in range(num_obstacles):
                w, h = self.np_random.integers(2, 5), self.np_random.integers(2, 5)
                x, y = self.np_random.integers(2, self.GRID_WIDTH - w - 2), self.np_random.integers(2, self.GRID_HEIGHT - h - 2)
                for i in range(w):
                    for j in range(h):
                        self.walls.add((x + i, y + j))

            # Place player and objective
            self.player_pos = (2, self.GRID_HEIGHT // 2)
            self.objective_pos = (self.GRID_WIDTH - 3, self.GRID_HEIGHT // 2)

            if self.player_pos in self.walls: self.walls.remove(self.player_pos)
            if self.objective_pos in self.walls: self.walls.remove(self.objective_pos)

            # Check for solvability using BFS
            q = deque([self.player_pos])
            visited = {self.player_pos}
            found = False
            while q:
                x, y = q.popleft()
                if (x, y) == self.objective_pos:
                    found = True
                    break
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in visited and (nx, ny) not in self.walls:
                        visited.add((nx, ny))
                        q.append((nx, ny))
            
            if found:
                break # Level is solvable

        # Generate enemy
        enemy_start_pos = (self.GRID_WIDTH // 2, 3)
        while enemy_start_pos in self.walls:
            enemy_start_pos = (self.np_random.integers(2, self.GRID_WIDTH - 2), self.np_random.integers(2, self.GRID_HEIGHT-2))
        
        self.enemies = [{
            "pos": enemy_start_pos,
            "path": [enemy_start_pos, (enemy_start_pos[0], self.GRID_HEIGHT - 4)],
            "path_index": 0,
            "direction_vec": np.array([0, 1.0]),
            "distracted_by": None, # (x,y) of clone
            "move_cooldown": 0
        }]

    def _handle_input(self, movement, space_pressed, shift_pressed):
        reward = 0
        
        # --- Movement ---
        px, py = self.player_pos
        if movement == 1: py -= 1 # Up
        elif movement == 2: py += 1 # Down
        elif movement == 3: px -= 1 # Left
        elif movement == 4: px += 1 # Right
        
        if (px, py) not in self.walls:
            self.player_pos = (px, py)
            # Create movement particles
            self._create_particles(self._grid_to_pixel(self.player_pos), 5, self.COLOR_PLAYER, 0.5)

        # --- Portal Teleportation ---
        if len(self.portals) == 2:
            p1, p2 = self.portals[0], self.portals[1]
            if self.player_pos == p1:
                self.player_pos = p2
                # Sound: Portal whoosh
                reward += 10.0 # Reward for using portal
                self._create_particles(self._grid_to_pixel(p1), 20, self.COLOR_PORTAL_1, 2.0)
                self._create_particles(self._grid_to_pixel(p2), 20, self.COLOR_PORTAL_2, 2.0)
            elif self.player_pos == p2:
                self.player_pos = p1
                # Sound: Portal whoosh
                reward += 10.0 # Reward for using portal
                self._create_particles(self._grid_to_pixel(p2), 20, self.COLOR_PORTAL_2, 2.0)
                self._create_particles(self._grid_to_pixel(p1), 20, self.COLOR_PORTAL_1, 2.0)

        # --- Clone Deployment ---
        if space_pressed:
            # Sound: Clone deployed
            self.clones.append({"pos": self.player_pos, "timer": self.CLONE_LIFESPAN})
            self._create_particles(self._grid_to_pixel(self.player_pos), 20, self.COLOR_CLONE, 1.5)

        # --- Portal Placement ---
        if shift_pressed:
            # Sound: Portal placed
            if self.player_pos not in self.portals:
                if len(self.portals) < 2:
                    self.portals.append(self.player_pos)
                else:
                    self.portals.pop(0)
                    self.portals.append(self.player_pos)
                
                color = self.COLOR_PORTAL_1 if len(self.portals) == 1 else self.COLOR_PORTAL_2
                self._create_particles(self._grid_to_pixel(self.player_pos), 20, color, 1.5)
        
        return reward

    def _update_clones(self):
        distracted_an_enemy = False
        surviving_clones = []
        for clone in self.clones:
            clone["timer"] -= 1
            if clone["timer"] > 0:
                surviving_clones.append(clone)
            else:
                # Sound: Clone fizzle out
                self._create_particles(self._grid_to_pixel(clone['pos']), 10, self.COLOR_CLONE, 0.8)
        self.clones = surviving_clones
        
        # Check if any clone distracts an enemy
        for enemy in self.enemies:
            is_newly_distracted = False
            enemy['distracted_by'] = None
            for clone in self.clones:
                in_cone, _ = self._is_in_cone(clone['pos'], enemy['pos'], enemy['direction_vec'], self.ENEMY_CONE_ANGLE, self.ENEMY_SIGHT_RANGE)
                if in_cone:
                    if enemy['distracted_by'] is None:
                        is_newly_distracted = True
                    enemy['distracted_by'] = clone['pos']
                    break # First clone in cone gets attention
            if is_newly_distracted:
                distracted_an_enemy = True
        return distracted_an_enemy

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy['move_cooldown'] > 0:
                enemy['move_cooldown'] -= 1
                continue

            if enemy['distracted_by']:
                # Face the distraction
                distraction_vec = np.array(enemy['distracted_by']) - np.array(enemy['pos'])
                norm = np.linalg.norm(distraction_vec)
                if norm > 0:
                    enemy['direction_vec'] = distraction_vec / norm
            else:
                # Move along path
                target_pos = enemy['path'][enemy['path_index']]
                if enemy['pos'] == target_pos:
                    enemy['path_index'] = (enemy['path_index'] + 1) % len(enemy['path'])
                    target_pos = enemy['path'][enemy['path_index']]
                
                ex, ey = enemy['pos']
                tx, ty = target_pos
                
                move_vec = np.array([0,0])
                if tx > ex: move_vec[0] = 1
                elif tx < ex: move_vec[0] = -1
                if ty > ey: move_vec[1] = 1
                elif ty < ey: move_vec[1] = -1
                
                new_pos = (ex + move_vec[0], ey + move_vec[1])
                if new_pos not in self.walls:
                    enemy['pos'] = new_pos
                    norm = np.linalg.norm(move_vec)
                    if norm > 0:
                        enemy['direction_vec'] = move_vec / norm
                
                enemy['move_cooldown'] = 1 # Move every 2 steps

    def _check_player_detection(self):
        for enemy in self.enemies:
            if enemy['distracted_by']:
                continue # Enemy is looking at a clone
            
            in_cone, _ = self._is_in_cone(self.player_pos, enemy['pos'], enemy['direction_vec'], self.ENEMY_CONE_ANGLE, self.ENEMY_SIGHT_RANGE)
            if in_cone:
                # Line of sight check
                is_occluded = False
                gx1, gy1 = self.player_pos
                gx2, gy2 = enemy['pos']
                for wall in self.walls:
                    if self._line_intersects_cell(gx1, gy1, gx2, gy2, wall[0], wall[1]):
                        is_occluded = True
                        break
                if not is_occluded:
                    return True, True # Detected
        
        # Check if in cone but not detected (e.g., enemy distracted)
        for enemy in self.enemies:
            in_cone, _ = self._is_in_cone(self.player_pos, enemy['pos'], enemy['direction_vec'], self.ENEMY_CONE_ANGLE, self.ENEMY_SIGHT_RANGE)
            if in_cone:
                return False, True
                
        return False, False # Safe

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['life'] -= 1

    # --- Rendering Methods ---

    def _render_game(self):
        self._render_background()
        self._render_walls()
        self._render_objective()
        self._render_portals()
        self._render_clones()
        self._render_enemies()
        self._render_player()
        self._render_particles()

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        
        # Rhythm Bar
        pulse = (1 + math.sin(self.steps * 0.2)) / 2
        bar_width = int(pulse * self.SCREEN_WIDTH * 0.8)
        bar_color = self.COLOR_PLAYER
        pygame.draw.rect(self.screen, bar_color, (self.SCREEN_WIDTH/2 - bar_width/2, self.SCREEN_HEIGHT - 10, bar_width, 5))

    def _render_walls(self):
        for x, y in self.walls:
            rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
            pygame.draw.rect(self.screen, self.COLOR_WALL_GLOW, rect, 2)
    
    def _render_objective(self):
        pos_px = self._grid_to_pixel(self.objective_pos)
        self._draw_glow(pos_px, self.GRID_SIZE * 1.5, self.COLOR_OBJECTIVE_GLOW)
        pygame.gfxdraw.box(self.screen, pygame.Rect(pos_px[0] - self.GRID_SIZE//2, pos_px[1] - self.GRID_SIZE//2, self.GRID_SIZE, self.GRID_SIZE), self.COLOR_OBJECTIVE)
        pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], self.GRID_SIZE//4, self.COLOR_BG)

    def _render_portals(self):
        for i, pos in enumerate(self.portals):
            pos_px = self._grid_to_pixel(pos)
            color = self.COLOR_PORTAL_1 if i == 0 else self.COLOR_PORTAL_2
            
            for j in range(5):
                alpha = 150 - j * 30
                radius_pulse = (math.sin(self.steps * 0.1 + j * 0.5) + 1) / 2
                radius = int((self.GRID_SIZE * 0.5 + radius_pulse * 5) * (1 - j*0.1))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], radius, (*color, alpha))
                    pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, (*color, alpha))

    def _render_clones(self):
        for clone in self.clones:
            pos_px = self._grid_to_pixel(clone['pos'])
            pulse = clone['timer'] / self.CLONE_LIFESPAN
            radius = int(self.GRID_SIZE * 0.4 * pulse)
            self._draw_glow(pos_px, self.GRID_SIZE * pulse, self.COLOR_CLONE_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], radius, self.COLOR_CLONE)
            pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, self.COLOR_CLONE)

    def _render_enemies(self):
        for enemy in self.enemies:
            # Render detection cone
            pos_px = self._grid_to_pixel(enemy['pos'])
            angle = math.atan2(enemy['direction_vec'][1], enemy['direction_vec'][0])
            cone_color = self.COLOR_DISTRACTED_CONE if enemy['distracted_by'] else self.COLOR_DETECTION_CONE
            
            p1 = pos_px
            p2 = (pos_px[0] + self.ENEMY_SIGHT_RANGE * self.GRID_SIZE * math.cos(angle - self.ENEMY_CONE_ANGLE / 2),
                  pos_px[1] + self.ENEMY_SIGHT_RANGE * self.GRID_SIZE * math.sin(angle - self.ENEMY_CONE_ANGLE / 2))
            p3 = (pos_px[0] + self.ENEMY_SIGHT_RANGE * self.GRID_SIZE * math.cos(angle + self.ENEMY_CONE_ANGLE / 2),
                  pos_px[1] + self.ENEMY_SIGHT_RANGE * self.GRID_SIZE * math.sin(angle + self.ENEMY_CONE_ANGLE / 2))
            
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), cone_color)

            # Render enemy body
            self._draw_glow(pos_px, self.GRID_SIZE * 0.8, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], self.GRID_SIZE // 3, self.COLOR_ENEMY)

    def _render_player(self):
        pos_px = self._grid_to_pixel(self.player_pos)
        self._draw_glow(pos_px, self.GRID_SIZE, self.COLOR_PLAYER_GLOW)
        poly_pts = [
            (pos_px[0], pos_px[1] - self.GRID_SIZE // 2),
            (pos_px[0] - self.GRID_SIZE // 3, pos_px[1] + self.GRID_SIZE // 3),
            (pos_px[0] + self.GRID_SIZE // 3, pos_px[1] + self.GRID_SIZE // 3)
        ]
        pygame.gfxdraw.aapolygon(self.screen, poly_pts, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, poly_pts, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 10))
        self.screen.blit(steps_text, (15, 30))

        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            status_text = "OBJECTIVE REACHED" if self.game_won else "AGENT DETECTED"
            color = self.COLOR_PLAYER if self.game_won else self.COLOR_ENEMY
            
            game_over_render = self.font_game_over.render(status_text, True, color)
            text_rect = game_over_render.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(game_over_render, text_rect)

    # --- Utility Methods ---

    def _grid_to_pixel(self, grid_pos):
        x, y = grid_pos
        return (x * self.GRID_SIZE + self.GRID_SIZE // 2, y * self.GRID_SIZE + self.GRID_SIZE // 2)

    def _get_manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_in_cone(self, target_pos, cone_origin_pos, cone_dir_vec, cone_angle_rad, cone_range):
        target_vec = np.array(target_pos) - np.array(cone_origin_pos)
        dist = np.linalg.norm(target_vec)

        if dist == 0 or dist > cone_range:
            return False, 0

        target_vec_norm = target_vec / dist
        dot_product = np.dot(target_vec_norm, cone_dir_vec)
        
        angle_threshold = math.cos(cone_angle_rad / 2)
        return dot_product > angle_threshold, dist

    def _line_intersects_cell(self, x1, y1, x2, y2, cx, cy):
        # Simple check for line of sight occlusion by a grid cell
        # This is a simplification and not perfectly accurate but works for this grid game
        for i in np.linspace(0, 1, 10):
            px = x1 + (x2 - x1) * i
            py = y1 + (y2 - y1) * i
            if int(px) == cx and int(py) == cy:
                return True
        return False

    def _draw_glow(self, pos, radius, color):
        step = -max(1, int(radius / 10))
        for i in range(int(radius), 0, step):
            alpha = color[3] * (1 - i / radius)**2
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i, (*color[:3], int(alpha)))
    
    def _create_particles(self, pos, count, color, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.0) * speed_mult
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(10, 25)
            self.particles.append({
                "pos": list(pos), "vel": vel, "life": life, "max_life": life,
                "color": color, "size": self.np_random.uniform(1, 4)
            })

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    # --- Example Usage & Manual Play ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # The original code had a separate pygame.display.set_mode here.
    # To run headlessly, we should not do this unless render_mode="human"
    # For this fix, we assume headless operation is the default.
    # The test harness will handle rendering.
    
    # Example of running an episode with random actions
    print("Running a short episode with random actions...")
    done = False
    truncated = False
    total_reward = 0
    step_count = 0
    while not done and not truncated:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        if done or truncated:
            print(f"Episode finished after {step_count} steps.")
            print(f"Final Info: {info}")
            break
            
    print(f"\nGame Over! Final Score: {total_reward:.2f}, Steps: {env.steps}")
    env.close()