import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:59:27.424248
# Source Brief: brief_01409.md
# Brief Index: 1409
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "A physics-based puzzle game. Resize and guide falling objects into their target zones before time runs out."
    user_guide = "Use ↑ and ↓ arrow keys to select an object. Press space to resize the selected object and slow down time."
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TARGET_INCOMPLETE = (200, 50, 50, 100)
    COLOR_TARGET_COMPLETE = (50, 200, 50, 150)
    COLOR_SELECTED_GLOW = (255, 255, 100)
    COLOR_TIME_SLOW_OVERLAY = (255, 255, 0, 30)
    OBJECT_COLORS = [(66, 135, 245), (245, 66, 66), (66, 245, 135), (245, 150, 66)]

    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game Parameters
    MAX_STEPS = 1500
    BASE_TIME_LIMIT = 60.0  # seconds
    TIME_DECREMENT_PER_5_LEVELS = 5.0
    FPS = 30
    
    # Physics & World
    GRAVITY = -300.0
    WORLD_DAMPING = 0.98
    OBJECT_RESTITUTION = 0.6 # bounciness
    WORLD_BOUNDS = np.array([200, 200]) # The "floor" size in world units

    # Isometric Projection
    ISO_X_AXIS = np.array([0.866, -0.5]) * 30
    ISO_Y_AXIS = np.array([0.866, 0.5]) * 30
    ISO_Z_AXIS = np.array([0.0, -1.0]) * 30
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # Game State
        self.steps = 0
        self.score = 0
        self.level = 0
        self.time_left = 0.0
        self.game_over = False
        self.objects = []
        self.particles = []
        self.selected_object_idx = 0
        
        # Action state tracking for press vs. hold
        self.prev_movement_action = 0
        self.prev_space_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level += 1
        
        time_decrement = (self.level // 5) * self.TIME_DECREMENT_PER_5_LEVELS
        self.time_left = max(15.0, self.BASE_TIME_LIMIT - time_decrement)

        self._generate_level()
        self.selected_object_idx = 0 if self.objects else -1
        
        self.prev_movement_action = 0
        self.prev_space_held = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Actions ---
        # 1. Object Selection (on press)
        if movement in [1, 2] and movement != self.prev_movement_action:
            if self.objects:
                # sfx: ui_select
                if movement == 1: # Up
                    self.selected_object_idx = (self.selected_object_idx + 1) % len(self.objects)
                elif movement == 2: # Down
                    self.selected_object_idx = (self.selected_object_idx - 1 + len(self.objects)) % len(self.objects)
        
        # 2. Size Toggle (on press)
        if space_held and not self.prev_space_held:
            if self.selected_object_idx != -1:
                # sfx: object_resize
                obj = self.objects[self.selected_object_idx]
                obj['size_state'] = 1 - obj['size_state']
                self._create_particles(obj['pos'], obj['color'], 20)
        
        # 3. Time Manipulation (on hold)
        time_scale = 0.3 if space_held else 1.0
        
        self.prev_movement_action = movement
        self.prev_space_held = space_held
        
        # --- Update Game Logic ---
        dt = self.clock.tick(self.FPS) / 1000.0
        self._update_physics(dt * time_scale)
        self.time_left -= dt
        self.steps += 1
        
        # --- Calculate Reward & Termination ---
        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            if all(obj['is_placed'] for obj in self.objects):
                # sfx: level_complete
                reward += 100.0 # Win bonus
            else:
                # sfx: time_out
                reward -= 100.0 # Timeout penalty
            self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_level(self):
        self.objects.clear()
        self.particles.clear()
        num_objects = min(4, 1 + self.level // 3)

        occupied_starts = []
        occupied_targets = []

        for i in range(num_objects):
            # Generate non-overlapping start position
            while True:
                start_pos = self.np_random.uniform(low=-self.WORLD_BOUNDS * 0.8, high=self.WORLD_BOUNDS * 0.8)
                too_close = any(np.linalg.norm(start_pos - pos) < 60 for pos in occupied_starts)
                if not too_close:
                    occupied_starts.append(start_pos)
                    break
            
            # Generate non-overlapping target position
            while True:
                target_pos = self.np_random.uniform(low=-self.WORLD_BOUNDS * 0.6, high=self.WORLD_BOUNDS * 0.6)
                too_close = any(np.linalg.norm(target_pos - pos) < 60 for pos in occupied_targets)
                if not too_close:
                    occupied_targets.append(target_pos)
                    break
            
            size1 = self.np_random.uniform(15, 25)
            size2 = self.np_random.uniform(35, 45)

            obj = {
                'pos': np.array([start_pos[0], start_pos[1], self.np_random.uniform(100, 200)], dtype=float),
                'vel': self.np_random.uniform(low=-10, high=10, size=3).astype(float),
                'size_state': self.np_random.integers(0, 2),
                'sizes': [size1, size2],
                'masses': [size1**2, size2**2], # Using area for mass in 2D projection
                'color': self.OBJECT_COLORS[i % len(self.OBJECT_COLORS)],
                'target_pos': target_pos,
                'target_radius': (size1 + size2) / 2 * 1.2, # Target is average size
                'is_placed': False,
                'dist_to_target_prev': np.linalg.norm(start_pos - target_pos)
            }
            self.objects.append(obj)

    def _update_physics(self, dt):
        if dt == 0: return

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel'] * dt
            p['life'] -= dt

        # Update objects
        for i, obj in enumerate(self.objects):
            # Apply gravity
            obj['vel'][2] += self.GRAVITY * dt
            
            # Update position
            obj['pos'] += obj['vel'] * dt
            
            # Dampen velocity
            obj['vel'] *= self.WORLD_DAMPING

            # Floor collision
            if obj['pos'][2] < obj['sizes'][obj['size_state']]:
                obj['pos'][2] = obj['sizes'][obj['size_state']]
                obj['vel'][2] *= -self.OBJECT_RESTITUTION
                # sfx: bounce_soft

            # Wall collisions
            for axis in [0, 1]:
                if abs(obj['pos'][axis]) > self.WORLD_BOUNDS[axis] - obj['sizes'][obj['size_state']]:
                    obj['pos'][axis] = np.sign(obj['pos'][axis]) * (self.WORLD_BOUNDS[axis] - obj['sizes'][obj['size_state']])
                    obj['vel'][axis] *= -self.OBJECT_RESTITUTION
                    # sfx: bounce_hard

        # Object-object collisions
        for i in range(len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                obj1 = self.objects[i]
                obj2 = self.objects[j]
                
                size1 = obj1['sizes'][obj1['size_state']]
                size2 = obj2['sizes'][obj2['size_state']]
                
                delta_pos = obj1['pos'] - obj2['pos']
                dist = np.linalg.norm(delta_pos)
                
                if 0 < dist < size1 + size2:
                    # sfx: object_collide
                    overlap = (size1 + size2 - dist)
                    normal = delta_pos / dist
                    
                    # Resolve overlap
                    obj1['pos'] += normal * overlap * 0.5
                    obj2['pos'] -= normal * overlap * 0.5
                    
                    # Resolve velocity (impulse)
                    delta_vel = obj1['vel'] - obj2['vel']
                    impulse_j = -2 * np.dot(delta_vel, normal) / (1/obj1['masses'][obj1['size_state']] + 1/obj2['masses'][obj2['size_state']])
                    
                    obj1['vel'] += impulse_j * normal / obj1['masses'][obj1['size_state']]
                    obj2['vel'] -= impulse_j * normal / obj2['masses'][obj2['size_state']]

    def _calculate_reward(self):
        reward = 0.0
        all_placed = True
        
        for obj in self.objects:
            # Distance-based reward
            dist_to_target = np.linalg.norm(obj['pos'][:2] - obj['target_pos'])
            reward += (obj['dist_to_target_prev'] - dist_to_target) * 0.01
            obj['dist_to_target_prev'] = dist_to_target

            # Placement reward
            is_now_placed = dist_to_target < obj['target_radius'] and abs(obj['vel'][2]) < 1.0
            if is_now_placed and not obj['is_placed']:
                reward += 5.0 # Placed bonus
                # sfx: object_placed
            elif not is_now_placed and obj['is_placed']:
                reward -= 5.0 # Un-placed penalty
            obj['is_placed'] = is_now_placed
            
            if not obj['is_placed']:
                all_placed = False

        if all_placed and self.objects:
            reward += 100.0 # Win bonus is handled in step() for termination
            
        return reward

    def _check_termination(self):
        if self.time_left <= 0:
            return True
        if self.objects and all(obj['is_placed'] for obj in self.objects):
            return True
        return False
        
    def _iso_transform(self, pos_3d):
        screen_pos = self.ISO_X_AXIS * pos_3d[0] + self.ISO_Y_AXIS * pos_3d[1] + self.ISO_Z_AXIS * pos_3d[2]
        return screen_pos + np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 100])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(-10, 11):
            start_3d = np.array([-self.WORLD_BOUNDS[0], i * 20, 0])
            end_3d = np.array([self.WORLD_BOUNDS[0], i * 20, 0])
            pygame.draw.aaline(self.screen, self.COLOR_GRID, self._iso_transform(start_3d), self._iso_transform(end_3d))
            start_3d = np.array([i * 20, -self.WORLD_BOUNDS[1], 0])
            end_3d = np.array([i * 20, self.WORLD_BOUNDS[1], 0])
            pygame.draw.aaline(self.screen, self.COLOR_GRID, self._iso_transform(start_3d), self._iso_transform(end_3d))

        # Combine all renderable items and sort by depth
        render_queue = []
        for i, obj in enumerate(self.objects):
            render_queue.append({'type': 'target', 'obj': obj})
            render_queue.append({'type': 'shadow', 'obj': obj})
            render_queue.append({'type': 'object', 'obj': obj, 'idx': i})
        
        # Depth sort: higher Y in world space is further away
        render_queue.sort(key=lambda item: item['obj']['pos'][0] + item['obj']['pos'][1] - item['obj']['pos'][2]*0.1)

        for item in render_queue:
            obj = item['obj']
            size = obj['sizes'][obj['size_state']]
            
            if item['type'] == 'target':
                target_pos_3d = np.array([obj['target_pos'][0], obj['target_pos'][1], 1])
                screen_pos = self._iso_transform(target_pos_3d)
                color = self.COLOR_TARGET_COMPLETE if obj['is_placed'] else self.COLOR_TARGET_INCOMPLETE
                pygame.gfxdraw.filled_ellipse(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(obj['target_radius']), int(obj['target_radius'] * 0.5), color)
            
            elif item['type'] == 'shadow':
                shadow_pos_3d = np.array([obj['pos'][0], obj['pos'][1], 0])
                screen_pos = self._iso_transform(shadow_pos_3d)
                shadow_size = int(size * (1 - min(0.8, obj['pos'][2] / 400)))
                shadow_alpha = int(100 * (1 - min(0.8, obj['pos'][2] / 400)))
                if shadow_size > 0:
                    pygame.gfxdraw.filled_ellipse(self.screen, int(screen_pos[0]), int(screen_pos[1]), shadow_size, int(shadow_size * 0.5), (0,0,0,shadow_alpha))
            
            elif item['type'] == 'object':
                screen_pos = self._iso_transform(obj['pos'])
                color = obj['color']
                
                # Draw main body
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(size), color)
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(size), tuple(c*0.8 for c in color))
                
                # Draw highlight
                highlight_pos = screen_pos + np.array([size*0.2, -size*0.2])
                pygame.gfxdraw.filled_circle(self.screen, int(highlight_pos[0]), int(highlight_pos[1]), int(size*0.3), (255,255,255,100))

                # Draw selection glow
                if item['idx'] == self.selected_object_idx:
                    for i in range(4):
                        pygame.gfxdraw.aacircle(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(size) + i + 2, self.COLOR_SELECTED_GLOW)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, p['life'] / p['max_life'])
            color = (*p['color'], int(alpha * 255))
            pygame.draw.circle(self.screen, color, p['pos'], int(p['size'] * alpha))

        # Draw time slow effect
        if self.prev_space_held:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_TIME_SLOW_OVERLAY)
            self.screen.blit(overlay, (0, 0))

    def _render_ui(self):
        # Time display
        time_text = f"TIME: {max(0, self.time_left):.1f}"
        time_surf = self.font_main.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Score display
        score_text = f"SCORE: {self.score:.0f}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))

        # Level display
        level_text = f"LEVEL: {self.level}"
        level_surf = self.font_main.render(level_text, True, self.COLOR_TEXT)
        self.screen.blit(level_surf, (self.SCREEN_WIDTH // 2 - level_surf.get_width() // 2, 10))

        # Controls hint
        controls_text = "[↑/↓] Select  [SPACE] Resize/Slow Time"
        controls_surf = self.font_small.render(controls_text, True, self.COLOR_TEXT)
        self.screen.blit(controls_surf, (10, self.SCREEN_HEIGHT - controls_surf.get_height() - 5))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "level": self.level,
            "objects_placed": sum(1 for obj in self.objects if obj['is_placed']),
            "total_objects": len(self.objects)
        }

    def _create_particles(self, pos_3d, color, count):
        screen_pos = self._iso_transform(pos_3d)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(20, 80)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.uniform(0.3, 0.8)
            self.particles.append({
                'pos': screen_pos.copy(),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })
            
    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # Example usage
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    # Un-comment the following line to run with a display window
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    pygame.display.set_caption("Physics Puzzle Environment")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Action defaults
        movement = 0
        space_held = 0
        shift_held = 0
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print(f"--- Level {info['level']} ---")

        # Get keyboard state for continuous actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
            print(f"--- Starting Level {info['level']} ---")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()