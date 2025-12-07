import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:35:30.958789
# Source Brief: brief_01824.md
# Brief Index: 1824
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
        "Grow a cosmic tree by launching glowing buds towards targets. Use magnetic fields to guide your shots and unlock new types of buds to complete the level."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to aim your launch. Press space to fire a bud and hold shift to cycle between available bud types."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game Constants
        self.W, self.H = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500

        # Gymnasium Spaces
        self.observation_space = Box(low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.Font(None, 20)
            self.font_medium = pygame.font.Font(None, 28)
            self.font_large = pygame.font.Font(None, 48)
        except FileNotFoundError:
            print("Default font not found, using pygame's fallback.")
            self.font_small = pygame.font.SysFont(None, 20)
            self.font_medium = pygame.font.SysFont(None, 28)
            self.font_large = pygame.font.SysFont(None, 48)


        # Colors
        self.COLOR_BG_TOP = (10, 15, 30)
        self.COLOR_BG_BOTTOM = (30, 25, 50)
        self.COLOR_BRANCH = (140, 255, 150)
        self.COLOR_TARGET = (255, 215, 0)
        self.COLOR_TARGET_AURA = (255, 215, 0, 50)
        self.COLOR_TARGET_HIT = (100, 180, 100)
        self.COLOR_MAGNET = (100, 150, 255, 20)
        self.COLOR_LAUNCHER = (255, 100, 100)
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.BUD_COLORS = {
            "standard": (50, 200, 255),
            "splitter": (255, 100, 200)
        }

        # Game State Variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.branches = []
        self.launch_nodes = []
        self.targets = []
        self.magnetic_fields = []
        self.active_bud = None
        self.particles = []
        self.camera_zoom = 1.0
        self.camera_focus = np.array([self.W / 2, self.H / 2], dtype=float)
        
        self.launch_angle = 0.0
        self.num_launches_left = 0
        self.available_bud_types = []
        self.selected_bud_type_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.targets_to_unlock_next_bud = 0
        
        # A call to reset() is expected in the constructor.
        # However, to avoid duplicate initialization, we will rely on the
        # user to call reset() before starting an episode.
        # self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Tree state
        initial_trunk_start = (self.W / 2, self.H - 20)
        initial_trunk_end = (self.W / 2, self.H - 50)
        self.branches = [{'start': initial_trunk_start, 'end': initial_trunk_end}]
        self.launch_nodes = [initial_trunk_end]
        
        # Player state
        self.launch_angle = -math.pi / 2  # Straight up
        self.num_launches_left = 15
        self.available_bud_types = ["standard"]
        self.selected_bud_type_idx = 0
        self.targets_to_unlock_next_bud = 3

        # Action state
        self.prev_space_held = False
        self.prev_shift_held = False

        # Level state
        self._generate_level()
        self.active_bud = None
        self.particles = []
        
        # Camera state
        self.camera_zoom = 1.0
        self.camera_focus = np.array([self.W / 2, self.H / 2], dtype=float)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0.0
        
        # 1. Update active bud if it exists
        if self.active_bud:
            reward += self._update_bud()

        # 2. Handle player input if no bud is flying
        else:
            self._handle_player_input(action)
        
        # 3. Update particles
        self._update_particles()

        # 4. Check for termination conditions
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if all(t['hit'] for t in self.targets):
                reward += 100  # Win bonus
            else:
                reward -= 100 # Loss penalty

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        self.targets = []
        num_targets = 5
        for _ in range(num_targets):
            pos = (self.np_random.uniform(50, self.W - 50), self.np_random.uniform(50, self.H - 150))
            self.targets.append({'pos': pos, 'radius': 12, 'hit': False})

        self.magnetic_fields = []
        num_magnets = self.np_random.integers(2, 5)
        for _ in range(num_magnets):
            pos = (self.np_random.uniform(100, self.W - 100), self.np_random.uniform(100, self.H - 200))
            strength = self.np_random.uniform(50, 150) * self.np_random.choice([-1, 1])
            self.magnetic_fields.append({'pos': pos, 'radius': self.np_random.uniform(50, 80), 'strength': strength})

    def _handle_player_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Adjust angle
        angle_speed = 0.05
        # Movement codes: 0=No-op, 1=Up, 2=Down, 3=Left, 4=Right
        if movement == 1 or movement == 3: # Up/Left -> counter-clockwise
            self.launch_angle -= angle_speed 
        if movement == 2 or movement == 4: # Down/Right -> clockwise
            self.launch_angle += angle_speed
        self.launch_angle = self.launch_angle % (2 * math.pi)

        # Launch bud on button press
        if space_held and not self.prev_space_held and self.num_launches_left > 0:
            self._launch_bud()

        # Cycle bud type on button press
        if shift_held and not self.prev_shift_held:
            self.selected_bud_type_idx = (self.selected_bud_type_idx + 1) % len(self.available_bud_types)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _launch_bud(self):
        if not self.launch_nodes: return
        
        launch_pos = np.array(self.launch_nodes[-1], dtype=float)
        bud_speed = 7.0
        velocity = np.array([math.cos(self.launch_angle), math.sin(self.launch_angle)]) * bud_speed
        
        dist, _ = self._get_nearest_unhit_target(launch_pos)

        self.active_bud = {
            'pos': launch_pos,
            'vel': velocity,
            'type': self.available_bud_types[self.selected_bud_type_idx],
            'path': [tuple(launch_pos)],
            'last_dist_to_target': dist,
            'life': 150 # Max lifetime in steps
        }
        self.num_launches_left -= 1

    def _update_bud(self):
        bud = self.active_bud
        bud['life'] -= 1
        
        # Apply magnetic forces
        accel = np.zeros(2)
        for field in self.magnetic_fields:
            vec_to_field = np.array(field['pos']) - bud['pos']
            dist_sq = np.dot(vec_to_field, vec_to_field)
            if dist_sq > 1: # Avoid division by zero
                force_mag = field['strength'] / dist_sq
                accel += (vec_to_field / math.sqrt(dist_sq)) * force_mag
        
        bud['vel'] += accel
        # Speed limit
        speed = np.linalg.norm(bud['vel'])
        if speed > 10.0:
            bud['vel'] = bud['vel'] * (10.0 / speed)

        bud['pos'] += bud['vel']
        bud['path'].append(tuple(bud['pos']))

        # Calculate reward for getting closer to a target
        dist, _ = self._get_nearest_unhit_target(bud['pos'])
        reward = (bud['last_dist_to_target'] - dist) * 0.1
        bud['last_dist_to_target'] = dist

        # Check for landing conditions
        landed = False
        # 1. Hit a target
        for target in self.targets:
            if not target['hit'] and np.linalg.norm(bud['pos'] - np.array(target['pos'])) < target['radius']:
                self._land_bud(bud['pos'], target['pos'])
                target['hit'] = True
                reward += 25
                self._create_particles(target['pos'], self.COLOR_TARGET, 30)
                self._check_unlocks()
                landed = True
                break
        if landed: return reward

        # 2. Hit an existing branch
        for branch in self.branches:
            if self._dist_point_to_segment(bud['pos'], np.array(branch['start']), np.array(branch['end'])) < 5:
                self._land_bud(bud['pos'])
                reward += 5
                self._create_particles(bud['pos'], self.COLOR_BRANCH, 15)
                landed = True
                break
        if landed: return reward

        # 3. Out of bounds or lifetime expired
        if not (0 < bud['pos'][0] < self.W and 0 < bud['pos'][1] < self.H) or bud['life'] <= 0:
            self.active_bud = None # Fizzle
            
        return np.clip(reward, -10, 10)

    def _land_bud(self, landing_pos, target_pos=None):
        bud = self.active_bud
        if not self.launch_nodes:
            self.active_bud = None
            return

        start_node = np.array(self.launch_nodes[-1])
        
        # Find closest point on existing tree to land
        closest_point_on_tree = start_node
        min_dist = np.linalg.norm(landing_pos - start_node)

        for branch in self.branches:
            p, d = self._project_point_on_segment(landing_pos, np.array(branch['start']), np.array(branch['end']))
            if d < min_dist:
                min_dist = d
                closest_point_on_tree = p
        
        end_point = landing_pos
        if target_pos:
            end_point = np.array(target_pos)
        
        bud_type = bud['type']
        if bud_type == "standard":
            self.branches.append({'start': tuple(closest_point_on_tree), 'end': tuple(end_point)})
            self.launch_nodes.append(tuple(end_point))
        elif bud_type == "splitter":
            direction = end_point - closest_point_on_tree
            length = np.linalg.norm(direction)
            if length > 1e-6:
                norm_dir = direction / length
                
                angle1 = math.radians(30)
                rot1 = np.array([[math.cos(angle1), -math.sin(angle1)], [math.sin(angle1), math.cos(angle1)]])
                dir1 = np.dot(rot1, norm_dir) * length * 0.7
                end1 = closest_point_on_tree + dir1
                self.branches.append({'start': tuple(closest_point_on_tree), 'end': tuple(end1)})
                self.launch_nodes.append(tuple(end1))
                
                angle2 = math.radians(-30)
                rot2 = np.array([[math.cos(angle2), -math.sin(angle2)], [math.sin(angle2), math.cos(angle2)]])
                dir2 = np.dot(rot2, norm_dir) * length * 0.7
                end2 = closest_point_on_tree + dir2
                self.branches.append({'start': tuple(closest_point_on_tree), 'end': tuple(end2)})
                self.launch_nodes.append(tuple(end2))

        self.active_bud = None

    def _check_unlocks(self):
        targets_hit = sum(1 for t in self.targets if t['hit'])
        if targets_hit >= self.targets_to_unlock_next_bud:
            if "splitter" not in self.available_bud_types:
                self.available_bud_types.append("splitter")
                self.targets_to_unlock_next_bud += 3 # Next unlock is harder

    def _check_termination(self):
        win = all(t['hit'] for t in self.targets)
        loss = self.num_launches_left <= 0 and self.active_bud is None
        timeout = self.steps >= self.MAX_STEPS
        return win or loss or timeout

    def _get_observation(self):
        self._update_camera()
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _update_camera(self):
        if not self.branches:
            return

        all_points = [p for b in self.branches for p in (b['start'], b['end'])]
        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)

        tree_width = max(max_x - min_x, 50)
        tree_height = max(max_y - min_y, 50)
        
        target_zoom_x = (self.W * 0.8) / tree_width
        target_zoom_y = (self.H * 0.8) / tree_height
        target_zoom = min(target_zoom_x, target_zoom_y, 2.0) # Cap max zoom

        target_focus = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])

        # Smooth interpolation
        self.camera_zoom = self.camera_zoom * 0.95 + target_zoom * 0.05
        self.camera_focus = self.camera_focus * 0.95 + target_focus * 0.05

    def _world_to_screen(self, pos):
        p = (np.array(pos) - self.camera_focus) * self.camera_zoom + np.array([self.W / 2, self.H / 2])
        return int(p[0]), int(p[1])

    def _render_background(self):
        for y in range(self.H):
            ratio = y / self.H
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.W, y))
    
    def _render_game(self):
        # Magnetic fields
        for field in self.magnetic_fields:
            pos = self._world_to_screen(field['pos'])
            radius = int(field['radius'] * self.camera_zoom)
            if radius > 1:
                self._draw_glowing_circle(self.screen, pos, radius, self.COLOR_MAGNET)

        # Targets
        for target in self.targets:
            pos = self._world_to_screen(target['pos'])
            radius = int(target['radius'] * self.camera_zoom)
            color = self.COLOR_TARGET_HIT if target['hit'] else self.COLOR_TARGET
            aura = (*color, 50)
            if radius > 1:
                self._draw_glowing_circle(self.screen, pos, radius, aura)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

        # Branches
        for branch in self.branches:
            start_pos = self._world_to_screen(branch['start'])
            end_pos = self._world_to_screen(branch['end'])
            width = max(1, int(3 * self.camera_zoom))
            pygame.draw.line(self.screen, self.COLOR_BRANCH, start_pos, end_pos, width)

        # Launch trajectory preview
        if not self.active_bud and self.launch_nodes:
            start_pos = np.array(self.launch_nodes[-1], dtype=float)
            vel = np.array([math.cos(self.launch_angle), math.sin(self.launch_angle)]) * 7.0
            
            points = []
            current_pos = start_pos.copy()
            for _ in range(50):
                accel = np.zeros(2)
                for field in self.magnetic_fields:
                    vec = np.array(field['pos']) - current_pos
                    dist_sq = np.dot(vec, vec)
                    if dist_sq > 1:
                        force_mag = field['strength'] / dist_sq
                        accel += (vec / math.sqrt(dist_sq)) * force_mag
                vel += accel
                speed = np.linalg.norm(vel)
                if speed > 10.0: vel *= (10.0 / speed)
                current_pos += vel
                points.append(self._world_to_screen(current_pos))

            if len(points) > 1:
                pygame.draw.aalines(self.screen, self.COLOR_LAUNCHER, False, points)

        # Active bud
        if self.active_bud:
            pos = self._world_to_screen(self.active_bud['pos'])
            color = self.BUD_COLORS[self.active_bud['type']]
            radius = max(2, int(5 * self.camera_zoom))
            self._draw_glowing_circle(self.screen, pos, radius, (*color, 100))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        # Particles
        for p in self.particles:
            pos = self._world_to_screen(p['pos'])
            radius = int(p['size'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p['color'])
        
        # Launcher indicator
        if not self.active_bud and self.launch_nodes:
            pos = self._world_to_screen(self.launch_nodes[-1])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(2, int(5*self.camera_zoom)), self.COLOR_LAUNCHER)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], max(2, int(5*self.camera_zoom)), self.COLOR_LAUNCHER)


    def _render_ui(self):
        # Launches left
        text = f"Launches: {self.num_launches_left}"
        surf = self.font_medium.render(text, True, self.COLOR_UI_TEXT)
        self.screen.blit(surf, (10, 10))

        # Targets hit
        targets_hit = sum(1 for t in self.targets if t['hit'])
        total_targets = len(self.targets)
        text = f"Targets: {targets_hit} / {total_targets}"
        surf = self.font_medium.render(text, True, self.COLOR_UI_TEXT)
        self.screen.blit(surf, (10, 40))
        
        # Current bud type
        bud_name = self.available_bud_types[self.selected_bud_type_idx].capitalize()
        bud_color = self.BUD_COLORS[self.available_bud_types[self.selected_bud_type_idx]]
        text = f"Bud: {bud_name}"
        surf = self.font_medium.render(text, True, bud_color)
        self.screen.blit(surf, (10, 70))
        
        # Game Over message
        if self.game_over:
            win = all(t['hit'] for t in self.targets)
            msg = "SUCCESS" if win else "TRY AGAIN"
            color = self.COLOR_TARGET if win else self.COLOR_LAUNCHER
            surf = self.font_large.render(msg, True, color)
            self.screen.blit(surf, (self.W/2 - surf.get_width()/2, self.H/2 - surf.get_height()/2))

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': (*color, self.np_random.integers(100, 255)),
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1
            p['size'] *= 0.95
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0.5]

    def _draw_glowing_circle(self, surface, pos, radius, color):
        if radius < 1: return
        max_glow = radius * 2
        for i in range(int(max_glow), int(radius), -1):
            alpha = int(color[3] * (1 - (i - radius) / (max_glow - radius))**2)
            if alpha > 0:
                glow_color = (*color[:3], alpha)
                pygame.gfxdraw.aacircle(surface, pos[0], pos[1], i, glow_color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "launches_left": self.num_launches_left,
            "targets_hit": sum(1 for t in self.targets if t['hit']),
            "game_over": self.game_over
        }

    def _get_nearest_unhit_target(self, pos):
        unhit_targets = [t for t in self.targets if not t['hit']]
        if not unhit_targets:
            return 0.0, None
        
        distances = [np.linalg.norm(np.array(t['pos']) - pos) for t in unhit_targets]
        min_dist_idx = np.argmin(distances)
        return distances[min_dist_idx], unhit_targets[min_dist_idx]

    @staticmethod
    def _dist_point_to_segment(p, v, w):
        l2 = np.dot(v - w, v - w)
        if l2 == 0.0: return np.linalg.norm(p - v)
        t = max(0, min(1, np.dot(p - v, w - v) / l2))
        projection = v + t * (w - v)
        return np.linalg.norm(p - projection)
    
    @staticmethod
    def _project_point_on_segment(p, v, w):
        l2 = np.dot(v - w, v - w)
        if l2 == 0.0: return v, np.linalg.norm(p-v)
        t = max(0, min(1, np.dot(p - v, w - v) / l2))
        projection = v + t * (w - v)
        return projection, np.linalg.norm(p - projection)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Magnetic Buds")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    # Action state for manual play
    movement = 0 # 0: none, 1: up, 2: down, 3: left, 4: right
    space_held = 0
    shift_held = 0
    
    # Use a set to track held keys for smoother controls
    held_keys = set()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                held_keys.add(event.key)
                if event.key == pygame.K_SPACE: space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
                    held_keys.clear()
                elif event.key == pygame.K_ESCAPE:
                    running = False

            if event.type == pygame.KEYUP:
                held_keys.discard(event.key)
                if event.key == pygame.K_SPACE: space_held = 0
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 0

        # Determine movement from held keys
        movement = 0
        if pygame.K_w in held_keys or pygame.K_UP in held_keys: movement = 1
        elif pygame.K_s in held_keys or pygame.K_DOWN in held_keys: movement = 2
        elif pygame.K_a in held_keys or pygame.K_LEFT in held_keys: movement = 3
        elif pygame.K_d in held_keys or pygame.K_RIGHT in held_keys: movement = 4
        
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to reset.")
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()