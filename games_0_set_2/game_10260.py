import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:05:21.349024
# Source Brief: brief_00260.md
# Brief Index: 260
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Dreamscape Stealth Environment
    The player must collect all green 'identity fragments' while avoiding red 'guardians'.
    The player can create purple 'illusions' to distract guardians.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}
    
    game_description = (
        "Collect all identity fragments while avoiding guardians. Create illusions to distract your pursuers."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to create an illusion and shift to cycle illusion types."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array", level=1):
        super().__init__()
        
        # --- Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.INITIAL_LEVEL = level

        # --- Colors ---
        self.COLOR_BG = (26, 26, 46) # #1a1a2e
        self.COLOR_PLAYER = (0, 255, 255) # Cyan
        self.COLOR_GUARDIAN = (255, 0, 128) # Magenta
        self.COLOR_FRAGMENT = (0, 255, 0) # Green
        self.ILLUSION_COLORS = [
            (255, 0, 255), # Purple
            (255, 165, 0), # Orange
            (255, 255, 0), # Yellow
            (138, 43, 226) # BlueViolet
        ]
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_PATH = (40, 40, 60)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_main = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 16)
        if self.render_mode == "human":
            pygame.display.set_caption("Dreamscape")
            self.display_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = self.INITIAL_LEVEL
        
        self.player_pos = None
        self.player_target_pos = None
        self.player_speed = 0.15

        self.fragments = []
        self.total_fragments = 0
        self.collected_fragments = 0

        self.guardians = []
        self.illusions = []
        self.particles = []

        self.selected_illusion_idx = 0
        self.last_space_held = 0
        self.last_shift_held = 0

        self.last_dist_to_fragment = 0
        self.last_dist_to_guardian = 0
        self.event_reward = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and 'level' in options:
            self.level = options['level']
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._setup_level()

        self.last_dist_to_fragment = self._get_min_dist_to_entity(self.player_pos, self.fragments)
        self.last_dist_to_guardian = self._get_min_dist_to_entity(self.player_pos, [g['pos'] for g in self.guardians])
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), self._get_info()

    def _setup_level(self):
        self.fragments = []
        self.guardians = []
        self.illusions = []
        self.particles = []
        self.collected_fragments = 0
        self.selected_illusion_idx = 0
        self.last_space_held = 0
        self.last_shift_held = 0

        # Place player
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_target_pos = self.player_pos.copy()

        # Generate Fragments
        self.total_fragments = min(3 + self.level, 8)
        while len(self.fragments) < self.total_fragments:
            pos = pygame.Vector2(
                self.np_random.integers(40, self.WIDTH - 40),
                self.np_random.integers(40, self.HEIGHT - 40)
            )
            if self._get_min_dist_to_entity(pos, self.fragments + [self.player_pos]) > 50:
                self.fragments.append(pos)

        # Generate Guardians
        num_guardians = min(1 + (self.level - 1), 5)
        for _ in range(num_guardians):
            path = []
            path_len = self.np_random.integers(3, 6)
            start_pos = pygame.Vector2(
                self.np_random.integers(20, self.WIDTH - 20),
                self.np_random.integers(20, self.HEIGHT - 20)
            )
            path.append(start_pos)
            for _ in range(path_len - 1):
                next_pos = pygame.Vector2(
                    self.np_random.integers(20, self.WIDTH - 20),
                    self.np_random.integers(20, self.HEIGHT - 20)
                )
                path.append(next_pos)
            
            self.guardians.append({
                'pos': start_pos.copy(),
                'target_pos': path[1].copy(),
                'path': path,
                'path_idx': 0,
                'speed': self.np_random.uniform(0.03, 0.06),
                'state': 'patrolling', # patrolling, distracted
                'distraction_timer': 0,
                'distraction_target': None,
                'vision_range': 120,
                'vision_angle': 45,
            })

    def step(self, action):
        self.steps += 1
        self.event_reward = 0
        
        movement, space_held, shift_held = action[0], action[1], action[2]
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        self._handle_input(movement, space_pressed, shift_pressed)
        self._update_player()
        self._update_guardians()
        self._update_illusions_and_particles()
        self._check_events()

        reward = self._calculate_reward()
        self.score += reward
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        if self.render_mode == "human":
            self._render_frame()

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_pressed):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        if move_vec.length() > 0:
            self.player_target_pos += move_vec * 30
            self.player_target_pos.x = np.clip(self.player_target_pos.x, 10, self.WIDTH - 10)
            self.player_target_pos.y = np.clip(self.player_target_pos.y, 10, self.HEIGHT - 10)

        if shift_pressed:
            self.selected_illusion_idx = (self.selected_illusion_idx + 1) % len(self.ILLUSION_COLORS)
            # sfx: UI_CYCLE

        if space_pressed:
            self.illusions.append({
                'pos': self.player_pos.copy(),
                'timer': 10 * self.metadata['render_fps'], # 10 seconds
                'color': self.ILLUSION_COLORS[self.selected_illusion_idx],
                'radius': 150
            })
            # sfx: ILLUSION_CAST

    def _update_player(self):
        self.player_pos.x = pygame.math.lerp(self.player_pos.x, self.player_target_pos.x, self.player_speed)
        self.player_pos.y = pygame.math.lerp(self.player_pos.y, self.player_target_pos.y, self.player_speed)

    def _update_guardians(self):
        for g in self.guardians:
            if g['state'] == 'distracted':
                g['distraction_timer'] -= 1
                if g['distraction_timer'] <= 0:
                    g['state'] = 'patrolling'
                    # Find nearest point on patrol path to resume
                    distances = [g['pos'].distance_to(p) for p in g['path']]
                    g['path_idx'] = np.argmin(distances)
                    next_idx = (g['path_idx'] + 1) % len(g['path'])
                    g['target_pos'] = g['path'][next_idx]
                else:
                    g['target_pos'] = g['distraction_target']
            
            elif g['state'] == 'patrolling':
                if g['pos'].distance_to(g['target_pos']) < 5:
                    g['path_idx'] = (g['path_idx'] + 1) % len(g['path'])
                    next_idx = (g['path_idx'] + 1) % len(g['path'])
                    g['target_pos'] = g['path'][next_idx]

            # Movement
            direction = (g['target_pos'] - g['pos']).normalize() if g['target_pos'] != g['pos'] else pygame.Vector2(0,0)
            g['pos'] += direction * g['speed'] * 30 # Scale speed with FPS

    def _update_illusions_and_particles(self):
        # Update illusions
        self.illusions = [i for i in self.illusions if i['timer'] > 0]
        for i in self.illusions:
            i['timer'] -= 1
            if self.np_random.random() < 0.5: # Particle spawn rate
                self.particles.append({
                    'pos': i['pos'].copy() + pygame.Vector2(self.np_random.uniform(-5, 5), 0),
                    'vel': pygame.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-2, -1)),
                    'timer': self.np_random.integers(20, 40),
                    'size': self.np_random.uniform(3, 6),
                    'color': i['color']
                })

        # Update particles
        self.particles = [p for p in self.particles if p['timer'] > 0]
        for p in self.particles:
            p['timer'] -= 1
            p['pos'] += p['vel']
            p['size'] *= 0.98

    def _check_events(self):
        # Guardian logic: distraction and capture
        for g in self.guardians:
            direction = (g['target_pos'] - g['pos']).normalize() if g['target_pos'] != g['pos'] else pygame.Vector2(1,0)
            
            # Check for player in vision cone
            if self._is_in_cone(self.player_pos, g['pos'], direction, g['vision_range'], g['vision_angle']):
                self.event_reward = -100
                self.game_over = True
                # sfx: PLAYER_CAUGHT
                return

            # Check for illusions in vision cone
            if g['state'] == 'patrolling':
                for illusion in self.illusions:
                    if self._is_in_cone(illusion['pos'], g['pos'], direction, g['vision_range'], g['vision_angle']):
                        g['state'] = 'distracted'
                        g['distraction_target'] = illusion['pos']
                        g['distraction_timer'] = 5 * self.metadata['render_fps'] # 5 seconds
                        self.event_reward += 1
                        # sfx: GUARDIAN_DISTRACTED
                        break # Only one distraction at a time

        # Player collects fragment
        for frag_pos in self.fragments[:]:
            if self.player_pos.distance_to(frag_pos) < 15:
                self.fragments.remove(frag_pos)
                self.collected_fragments += 1
                self.event_reward += 5
                # sfx: FRAGMENT_COLLECT
        
        # Win condition
        if not self.fragments:
            self.event_reward += 100
            self.game_over = True
            # sfx: LEVEL_COMPLETE

    def _calculate_reward(self):
        shaping_reward = 0
        
        # Distance to nearest fragment
        current_dist_to_fragment = self._get_min_dist_to_entity(self.player_pos, self.fragments)
        if current_dist_to_fragment is not None and self.last_dist_to_fragment is not None:
            shaping_reward += (self.last_dist_to_fragment - current_dist_to_fragment) * 0.01
        self.last_dist_to_fragment = current_dist_to_fragment

        # Distance to nearest non-distracted guardian
        non_distracted_guardians = [g['pos'] for g in self.guardians if g['state'] == 'patrolling']
        current_dist_to_guardian = self._get_min_dist_to_entity(self.player_pos, non_distracted_guardians)
        if current_dist_to_guardian is not None and self.last_dist_to_guardian is not None:
            # Reward for moving away from guardian
            shaping_reward += (current_dist_to_guardian - self.last_dist_to_guardian) * 0.01
        self.last_dist_to_guardian = current_dist_to_guardian

        return np.clip(self.event_reward + shaping_reward, -100, 100)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        if self.render_mode == "human":
            obs = self._get_observation()
            # The observation is (H, W, C), but pygame needs (W, H) surface.
            # _get_observation already renders to self.screen, so we just blit it.
            self.display_screen.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return self._get_observation()

    def _render_game(self):
        # Render patrol paths
        for g in self.guardians:
            if len(g['path']) > 1:
                pygame.draw.aalines(self.screen, self.COLOR_PATH, True, [p for p in g['path']], 1)

        # Render fragments
        for pos in self.fragments:
            pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
            radius = int(8 + pulse * 4)
            self._draw_glow(self.screen, self.COLOR_FRAGMENT, pos, radius, 4)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 6, self.COLOR_FRAGMENT)
        
        # Render illusions and particles
        for p in self.particles:
            alpha = int(255 * (p['timer'] / 40.0))
            color = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), color)
        
        # Render guardians
        for g in self.guardians:
            direction = (g['target_pos'] - g['pos']).normalize() if g['target_pos'] != g['pos'] else pygame.Vector2(1,0)
            self._draw_vision_cone(self.screen, g['pos'], direction, g['vision_range'], g['vision_angle'], g['state'])
            self._draw_glow(self.screen, self.COLOR_GUARDIAN, g['pos'], 15, 5)
            pygame.gfxdraw.filled_circle(self.screen, int(g['pos'].x), int(g['pos'].y), 8, self.COLOR_GUARDIAN)

        # Render player
        self._draw_glow(self.screen, self.COLOR_PLAYER, self.player_pos, 20, 6)
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), 10, self.COLOR_PLAYER)

    def _render_ui(self):
        # Fragments collected
        frag_text = self.font_main.render(f"FRAGMENTS: {self.collected_fragments}/{self.total_fragments}", True, self.COLOR_UI_TEXT)
        self.screen.blit(frag_text, (10, 10))
        
        # Selected illusion
        ui_box_size = 40
        padding = 5
        for i in range(len(self.ILLUSION_COLORS)):
            x = self.WIDTH - (len(self.ILLUSION_COLORS) - i) * (ui_box_size + padding)
            y = self.HEIGHT - ui_box_size - padding
            rect = pygame.Rect(x, y, ui_box_size, ui_box_size)
            
            if i == self.selected_illusion_idx:
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=3)
            else:
                pygame.draw.rect(self.screen, (100, 100, 120), rect, 1, border_radius=3)
            
            pygame.draw.rect(self.screen, self.ILLUSION_COLORS[i], rect.inflate(-8, -8), 0, border_radius=3)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "fragments_collected": self.collected_fragments,
            "game_over_reason": "win" if not self.fragments and self.game_over else ("loss" if self.game_over else "none")
        }

    def close(self):
        pygame.quit()

    # --- Helper Functions ---
    def _get_min_dist_to_entity(self, pos, entity_list):
        if not entity_list:
            return None
        return min([pos.distance_to(e) for e in entity_list])

    def _draw_glow(self, surface, color, center, max_radius, layers):
        for i in range(layers, 0, -1):
            radius = int(max_radius * (i / layers))
            alpha = int(100 * (1 - (i / layers))**2)
            try:
                glow_color = color + (alpha,)
            except TypeError:
                glow_color = tuple(list(color)[:3]) + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(center.x), int(center.y), radius, glow_color)

    def _is_in_cone(self, point, cone_pos, cone_dir, cone_range, cone_angle):
        vec_to_point = point - cone_pos
        dist = vec_to_point.length()
        if dist > cone_range or dist == 0:
            return False
        
        angle_to_point = math.degrees(math.acos(cone_dir.dot(vec_to_point.normalize())))
        return angle_to_point < cone_angle / 2

    def _draw_vision_cone(self, surface, pos, direction, range, angle, state):
        if state == 'distracted':
            color = self.ILLUSION_COLORS[0] + (20,)
        else:
            color = self.COLOR_GUARDIAN + (30,)

        angle_rad = math.radians(angle / 2)
        dir_angle = math.atan2(direction.y, direction.x)
        
        p1 = pos
        p2 = pos + pygame.Vector2(math.cos(dir_angle - angle_rad), math.sin(dir_angle - angle_rad)) * range
        p3 = pos + pygame.Vector2(math.cos(dir_angle + angle_rad), math.sin(dir_angle + angle_rad)) * range
        
        pygame.gfxdraw.aapolygon(surface, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)], color)
        pygame.gfxdraw.filled_polygon(surface, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)], color)


if __name__ == '__main__':
    # Example of how to use the environment
    env = GameEnv(render_mode="human")
    
    # --- Manual Control ---
    # Comment out the random agent loop to use manual control
    use_manual_control = True
    if use_manual_control:
        keys_to_action = {
            pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4
        }
        obs, info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            movement, space, shift = 0, 0, 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            keys = pygame.key.get_pressed()
            for key, move_action in keys_to_action.items():
                if keys[key]:
                    movement = move_action
            
            if keys[pygame.K_SPACE]:
                space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift = 1
                
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

    # --- Random Agent ---
    else:
        episodes = 5
        for ep in range(episodes):
            obs, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0
            step_count = 0
            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
            print(f"Episode {ep+1}: Total Reward={total_reward:.2f}, Steps={step_count}, Final Info={info}")

    env.close()