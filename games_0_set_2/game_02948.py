
# Generated: 2025-08-27T21:54:41.843608
# Source Brief: brief_02948.md
# Brief Index: 2948

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A top-down arcade game where the player hops between procedurally generated platforms.
    The goal is to reach the end platform while dodging enemies that patrol the level.
    The game is turn-based, with each action corresponding to a single hop.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to hop to the nearest platform in that direction. "
        "Hold Shift to hop to the absolute nearest platform. "
        "Press Space to hop to the furthest safe platform."
    )

    game_description = (
        "Hop between procedurally generated platforms, dodging enemies, to reach the end goal. "
        "Each hop is a turn. Plan your path carefully to maximize your score and reach the exit!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.ENEMY_SPEED_INITIAL = 0.5
        self.ENEMY_SPEED_INCREASE = 0.05
        self.ENEMY_SPEED_MAX = 2.0

        # --- Colors ---
        self.COLOR_BG = (44, 62, 80)
        self.COLOR_PLATFORM = (149, 165, 166)
        self.COLOR_GOAL = (46, 204, 113)
        self.COLOR_PLAYER = (241, 196, 15)
        self.COLOR_ENEMY = (231, 76, 60)
        self.COLOR_TEXT = (236, 240, 241)
        self.COLOR_PARTICLE = (255, 255, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.platforms = []
        self.platform_neighbors = {}
        self.start_platform_idx = 0
        self.end_platform_idx = 0
        self.player_platform_idx = 0
        self.enemies = []
        self.enemy_speed = self.ENEMY_SPEED_INITIAL
        self.particles = []
        self.last_hop_destination = (0, 0)
        self.last_reward_info = ""

        # --- Final Validation ---
        # self.validate_implementation() # Optional: call for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.enemy_speed = self.ENEMY_SPEED_INITIAL
        self.particles = []

        self._generate_level()
        self.player_platform_idx = self.start_platform_idx
        self._place_enemies()

        self.last_hop_destination = self.platforms[self.player_platform_idx].center
        self.last_reward_info = "Game Start!"

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        prev_platform_idx = self.player_platform_idx
        target_platform_idx = self._get_target_platform(movement, space_held, shift_held)

        reward = 0
        reward_info_parts = []

        if target_platform_idx is not None and target_platform_idx != self.player_platform_idx:
            self.player_platform_idx = target_platform_idx
            self.last_hop_destination = self.platforms[self.player_platform_idx].center
            self._create_particles(self.last_hop_destination, 20)
            # sfx: player_hop.wav

            # --- Calculate Reward ---
            reward += 0.1  # Base reward for moving
            reward_info_parts.append("+0.1 hop")

            # Distance-to-goal reward
            dist_prev = abs(self.platforms[prev_platform_idx].centerx - self.platforms[self.end_platform_idx].centerx)
            dist_new = abs(self.platforms[self.player_platform_idx].centerx - self.platforms[self.end_platform_idx].centerx)
            if dist_new < dist_prev:
                reward += 1.0
                reward_info_parts.append("+1.0 progress")
            else:
                reward -= 1.0
                reward_info_parts.append("-1.0 regress")

            # Risk reward for hopping near an enemy
            is_risky_hop = False
            for enemy in self.enemies:
                if self.player_platform_idx in self.platform_neighbors.get(enemy['platform_idx'], []):
                    is_risky_hop = True
                    break
            if is_risky_hop:
                reward += 0.2
                reward_info_parts.append("+0.2 risky")

        else: # No-op or invalid hop
            reward -= 0.1 # Small penalty for inaction
            reward_info_parts.append("-0.1 no-op")

        self._move_enemies()
        
        terminated = self._check_termination()

        if terminated:
            if self.player_platform_idx == self.end_platform_idx:
                reward += 100.0
                reward_info_parts = ["+100.0 GOAL!"]
                # sfx: win_game.wav
            else: # Collision or timeout
                is_collision = any(self.player_platform_idx == e['platform_idx'] for e in self.enemies)
                if is_collision:
                    reward -= 100.0
                    reward_info_parts = ["-100.0 COLLISION!"]
                    # sfx: player_death.wav
        
        self.last_reward_info = " | ".join(reward_info_parts)
        self.score += reward
        self.steps += 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 100 == 0:
            self.enemy_speed = min(self.ENEMY_SPEED_MAX, self.enemy_speed + self.ENEMY_SPEED_INCREASE)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_platforms()
        self._update_and_draw_particles()
        self._draw_enemies()
        self._draw_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _generate_level(self):
        self.platforms = []
        grid_w, grid_h = 12, 8
        cell_w, cell_h = self.WIDTH / grid_w, self.HEIGHT / grid_h
        min_plat_w, max_plat_w = int(cell_w * 0.6), int(cell_w * 0.9)
        min_plat_h, max_plat_h = int(cell_h * 0.4), int(cell_h * 0.6)

        # Create a main path
        path = []
        curr_y = self.np_random.integers(1, grid_h - 1)
        for x in range(grid_w):
            path.append((x, curr_y))
            if x < grid_w - 1:
                curr_y += self.np_random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
                curr_y = np.clip(curr_y, 1, grid_h - 2)
        
        platform_coords = set(path)

        # Add branches and random platforms
        for _ in range(20):
            ptype = self.np_random.random()
            if ptype < 0.5 and path: # Branch from path
                px, py = random.choice(path)
            else: # Random island
                px, py = self.np_random.integers(0, grid_w), self.np_random.integers(0, grid_h)

            for _ in range(self.np_random.integers(1, 4)):
                platform_coords.add((px, py))
                px += self.np_random.choice([-1, 1])
                py += self.np_random.choice([-1, 1])
                px = np.clip(px, 0, grid_w - 1)
                py = np.clip(py, 0, grid_h - 1)

        # Convert grid coords to rects
        coord_map = {}
        for i, (gx, gy) in enumerate(platform_coords):
            w = self.np_random.integers(min_plat_w, max_plat_w + 1)
            h = self.np_random.integers(min_plat_h, max_plat_h + 1)
            px = gx * cell_w + (cell_w - w) / 2 + self.np_random.uniform(-5, 5)
            py = gy * cell_h + (cell_h - h) / 2 + self.np_random.uniform(-5, 5)
            self.platforms.append(pygame.Rect(int(px), int(py), w, h))
            coord_map[(gx, gy)] = i

        self.start_platform_idx = coord_map[path[0]]
        self.end_platform_idx = coord_map[path[-1]]

        # Pre-calculate neighbors
        self.platform_neighbors = {i: [] for i in range(len(self.platforms))}
        for i, p1 in enumerate(self.platforms):
            for j, p2 in enumerate(self.platforms):
                if i == j: continue
                dist = math.hypot(p1.centerx - p2.centerx, p1.centery - p2.centery)
                if dist < cell_w * 1.8: # Connect if within ~1.8 grid cells
                    self.platform_neighbors[i].append(j)

    def _place_enemies(self):
        self.enemies = []
        num_enemies = self.np_random.integers(3, 6)
        
        possible_indices = list(range(len(self.platforms)))
        if self.start_platform_idx in possible_indices:
            possible_indices.remove(self.start_platform_idx)
        if self.end_platform_idx in possible_indices:
            possible_indices.remove(self.end_platform_idx)
        
        for _ in range(num_enemies):
            if not possible_indices: break
            
            start_idx = self.np_random.choice(possible_indices)
            possible_indices.remove(start_idx)
            
            patrol_partners = self.platform_neighbors.get(start_idx, [])
            if not patrol_partners: continue # Can't patrol if isolated
            
            end_idx = self.np_random.choice(patrol_partners)

            self.enemies.append({
                'platform_idx': start_idx,
                'path': (start_idx, end_idx),
                'progress': self.np_random.random(), # Start at random point on path
                'direction': 1,
            })

    def _get_target_platform(self, movement, space_held, shift_held):
        current_pos = self.platforms[self.player_platform_idx].center
        
        # Priority: Space > Shift > Movement
        if space_held: # Hop to furthest safe platform
            enemy_platforms = {e['platform_idx'] for e in self.enemies}
            safe_indices = [i for i, p in enumerate(self.platforms) if i != self.player_platform_idx and i not in enemy_platforms]
            if not safe_indices: return None
            
            distances = [math.hypot(p.centerx - current_pos[0], p.centery - current_pos[1]) for p in [self.platforms[i] for i in safe_indices]]
            return safe_indices[np.argmax(distances)]

        if shift_held: # Hop to nearest platform
            neighbor_indices = [i for i in range(len(self.platforms)) if i != self.player_platform_idx]
            if not neighbor_indices: return None
            
            distances = [math.hypot(p.centerx - current_pos[0], p.centery - current_pos[1]) for p in [self.platforms[i] for i in neighbor_indices]]
            return neighbor_indices[np.argmin(distances)]

        if movement > 0: # Directional hop
            candidates = []
            for i, p in enumerate(self.platforms):
                if i == self.player_platform_idx: continue
                dx, dy = p.centerx - current_pos[0], p.centery - current_pos[1]
                angle = math.atan2(-dy, dx) # -dy because pygame y is inverted
                
                # Up (pi/4 to 3pi/4)
                if movement == 1 and math.pi / 4 < angle < 3 * math.pi / 4: candidates.append(i)
                # Down (-3pi/4 to -pi/4)
                elif movement == 2 and -3 * math.pi / 4 < angle < -math.pi / 4: candidates.append(i)
                # Left (3pi/4 to 5pi/4)
                elif movement == 3 and (angle > 3 * math.pi / 4 or angle < -3 * math.pi / 4): candidates.append(i)
                # Right (-pi/4 to pi/4)
                elif movement == 4 and -math.pi / 4 < angle < math.pi / 4: candidates.append(i)

            if not candidates: return None
            distances = [math.hypot(self.platforms[i].centerx - current_pos[0], self.platforms[i].centery - current_pos[1]) for i in candidates]
            return candidates[np.argmin(distances)]

        return None # No-op

    def _move_enemies(self):
        for enemy in self.enemies:
            enemy['progress'] += enemy['direction'] * self.enemy_speed * (1/30.0) # Assume 30 FPS logic
            if not (0 <= enemy['progress'] <= 1):
                enemy['direction'] *= -1
                enemy['progress'] = np.clip(enemy['progress'], 0, 1)

            p1 = self.platforms[enemy['path'][0]].center
            p2 = self.platforms[enemy['path'][1]].center
            
            # Linear interpolation for position
            x = p1[0] + (p2[0] - p1[0]) * enemy['progress']
            y = p1[1] + (p2[1] - p1[1]) * enemy['progress']
            
            # Snap to platform when very close to an endpoint
            if enemy['progress'] < 0.05: enemy['platform_idx'] = enemy['path'][0]
            elif enemy['progress'] > 0.95: enemy['platform_idx'] = enemy['path'][1]
            else: enemy['platform_idx'] = -1 # In-transit, not on any platform

    def _check_termination(self):
        if self.player_platform_idx == self.end_platform_idx:
            self.game_over = True
            return True
        if any(self.player_platform_idx == e['platform_idx'] for e in self.enemies):
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _create_particles(self, position, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30) # Frames to live
            self.particles.append({'pos': list(position), 'vel': vel, 'life': life, 'max_life': life})

    def _update_and_draw_particles(self):
        remaining_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                remaining_particles.append(p)
                alpha = 255 * (p['life'] / p['max_life'])
                size = int(3 * (p['life'] / p['max_life']))
                if size > 0:
                    rect = pygame.Rect(int(p['pos'][0] - size/2), int(p['pos'][1] - size/2), size, size)
                    pygame.draw.rect(self.screen, self.COLOR_PARTICLE, rect)
        self.particles = remaining_particles

    def _draw_platforms(self):
        for i, p in enumerate(self.platforms):
            color = self.COLOR_GOAL if i == self.end_platform_idx else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, p, border_radius=3)

    def _draw_player(self):
        pos = self.platforms[self.player_platform_idx].center
        radius = 10
        # Glow effect
        glow_radius = int(radius * 1.8)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.COLOR_PLAYER, 60), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (pos[0] - glow_radius, pos[1] - glow_radius))
        # Player circle
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, pos, radius)

    def _draw_enemies(self):
        angle = (self.steps * 3) % 360 # Simple rotation animation
        rad_angle = math.radians(angle)
        size = 12
        for enemy in self.enemies:
            p1 = self.platforms[enemy['path'][0]].center
            p2 = self.platforms[enemy['path'][1]].center
            pos = [p1[i] + (p2[i] - p1[i]) * enemy['progress'] for i in (0, 1)]
            
            points = []
            for i in range(3):
                a = rad_angle + i * 2 * math.pi / 3
                x = pos[0] + size * math.cos(a)
                y = pos[1] + size * math.sin(a)
                points.append((int(x), int(y)))
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

    def _render_ui(self):
        score_text = self.font.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font.render(f"Step: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        if self.last_reward_info:
            reward_surf = self.small_font.render(self.last_reward_info, True, self.COLOR_TEXT)
            self.screen.blit(reward_surf, (10, 40))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Set up a window to display the game
    pygame.display.set_caption("Platform Hopper")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        env.clock.tick(10) # Limit manual play speed
        
    env.close()