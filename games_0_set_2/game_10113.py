import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:50:52.404281
# Source Brief: brief_00113.md
# Brief Index: 113
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for managing particle effects
class Particle:
    def __init__(self, x, y, color, size, life, dx=0, dy=0, gravity=0, shrink_rate=0.1):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.life = life
        self.initial_life = life
        self.dx = dx if dx != 0 else random.uniform(-1, 1)
        self.dy = dy if dy != 0 else random.uniform(-1, 1)
        self.gravity = gravity
        self.shrink_rate = shrink_rate

    def update(self):
        self.life -= 1
        self.x += self.dx
        self.y += self.dy
        self.dy += self.gravity
        self.size = max(0, self.size - self.shrink_rate)

    def draw(self, surface):
        if self.life > 0 and self.size > 0:
            alpha = int(255 * (self.life / self.initial_life))
            color = self.color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.size), color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    game_description = (
        "A stealth-based puzzle game. Evade patrolling guards, create illusions to distract them, and reach the exit."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to create an illusion and shift to toggle guard vision cones."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    TILE_SIZE = 40
    GRID_WIDTH, GRID_HEIGHT = SCREEN_WIDTH // TILE_SIZE, SCREEN_HEIGHT // TILE_SIZE
    MAX_STEPS = 2000
    FPS = 30

    # Colors
    COLOR_BG = (26, 26, 46)
    COLOR_WALL = (40, 40, 60)
    COLOR_PLAYER = (0, 168, 255)
    COLOR_GUARD = (255, 63, 63)
    COLOR_ILLUSION = (247, 183, 49)
    COLOR_EXIT = (76, 175, 80)
    COLOR_TEXT = (230, 230, 230)
    COLOR_UI_BG = (30, 30, 50, 180)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        if self.render_mode == "human":
            self.human_screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 18, bold=True)
        
        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_visual_pos = None
        self.guards = None
        self.illusion = None
        self.particles = None
        self.walls = None
        self.exit_pos = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.guard_speed = None
        self.show_vision_cones = None
        self.prev_shift_held = None
        self.prev_space_held = None

        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.guard_speed = 0.05  # Tiles per step

        # Player setup
        self.player_pos = np.array([1, self.GRID_HEIGHT // 2], dtype=float)
        self.player_visual_pos = self.player_pos.copy() * self.TILE_SIZE + self.TILE_SIZE / 2

        # Level layout
        self._generate_level()

        # Guards setup
        self.guards = self._initialize_guards()

        # Illusion and effects
        self.illusion = None
        self.particles = []
        
        # UI state
        self.show_vision_cones = False
        self.prev_shift_held = False
        self.prev_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward

        # --- 1. Handle Actions ---
        action_reward = self._handle_actions(action)
        reward += action_reward

        # --- 2. Update Game State ---
        self._update_player()
        distraction_reward = self._update_guards()
        reward += distraction_reward
        self._update_illusion()
        self._update_particles()
        
        # --- 3. Check for Events & Termination ---
        event_reward, terminated = self._check_events()
        reward += event_reward
        self.score += reward
        self.game_over = terminated

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        # --- 4. Difficulty Scaling ---
        if self.steps > 0 and self.steps % 500 == 0:
            self.guard_speed += 0.005 # Slower progression than brief for better gameplay

        # --- 5. Return Gymnasium Tuple ---
        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Movement
        target_pos = self.player_pos.copy()
        if movement == 1: target_pos[1] -= 1  # Up
        elif movement == 2: target_pos[1] += 1  # Down
        elif movement == 3: target_pos[0] -= 1  # Left
        elif movement == 4: target_pos[0] += 1  # Right
        
        if movement != 0 and not self._is_wall(int(target_pos[0]), int(target_pos[1])):
            self.player_pos = target_pos

        # Place Illusion (Space) - only on press, not hold
        if space_held and not self.prev_space_held and self.illusion is None:
            self.illusion = {
                "pos": self.player_pos.copy(),
                "duration": 150, # 5 seconds at 30fps
                "max_duration": 150,
                "newly_distracted": set()
            }
            # Sound: Illusion placed
            for _ in range(30):
                self.particles.append(Particle(
                    (self.player_pos[0] + 0.5) * self.TILE_SIZE,
                    (self.player_pos[1] + 0.5) * self.TILE_SIZE,
                    self.COLOR_ILLUSION, self.np_random.uniform(2, 6), 40,
                    dx=self.np_random.uniform(-2, 2), dy=self.np_random.uniform(-2, 2),
                    shrink_rate=0.1
                ))

        # Toggle Vision Cones (Shift) - only on press
        if shift_held and not self.prev_shift_held:
            self.show_vision_cones = not self.show_vision_cones
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        return reward

    def _update_player(self):
        target_visual_pos = self.player_pos * self.TILE_SIZE + self.TILE_SIZE / 2
        self.player_visual_pos += (target_visual_pos - self.player_visual_pos) * 0.5

    def _update_guards(self):
        distraction_reward = 0
        for guard in self.guards:
            # Determine target
            guard['is_distracted'] = False
            target_pos = None
            
            if self.illusion:
                illusion_pixel_pos = (self.illusion['pos'] + 0.5) * self.TILE_SIZE
                if self._is_in_vision_cone(guard, illusion_pixel_pos):
                    guard['is_distracted'] = True
                    target_pos = self.illusion['pos'].copy()
                    if guard['id'] not in self.illusion['newly_distracted']:
                        distraction_reward += 1.0
                        # Sound: Guard alert
                        self.illusion['newly_distracted'].add(guard['id'])

            if not guard['is_distracted']:
                target_pos = guard['patrol_path'][guard['patrol_index']]
                dist_to_waypoint = np.linalg.norm(guard['pos'] - target_pos)
                if dist_to_waypoint < 0.1:
                    guard['patrol_index'] = (guard['patrol_index'] + 1) % len(guard['patrol_path'])
                    target_pos = guard['patrol_path'][guard['patrol_index']]

            # Move towards target
            direction = target_pos - guard['pos']
            dist = np.linalg.norm(direction)
            if dist > 0:
                move_vec = direction / dist * self.guard_speed
                if dist < self.guard_speed:
                    guard['pos'] = target_pos.copy()
                else:
                    guard['pos'] += move_vec
                guard['facing_angle'] = math.atan2(move_vec[1], move_vec[0])

            # Update visual position
            target_visual_pos = guard['pos'] * self.TILE_SIZE + self.TILE_SIZE / 2
            guard['visual_pos'] += (target_visual_pos - guard['visual_pos']) * 0.5
        return distraction_reward

    def _update_illusion(self):
        if self.illusion:
            self.illusion['duration'] -= 1
            if self.illusion['duration'] <= 0:
                # Sound: Illusion dissipates
                self.illusion = None
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _check_events(self):
        reward = 0
        terminated = False

        # Check if caught
        player_center = self.player_pos + 0.5
        for guard in self.guards:
            guard_center = guard['pos'] + 0.5
            if np.linalg.norm(player_center - guard_center) * self.TILE_SIZE < self.TILE_SIZE * 0.8:
                reward = -5.0
                terminated = True
                # Sound: Player caught
                break
        
        # Check for win
        if np.linalg.norm(self.player_pos - self.exit_pos) < 0.5:
            reward = 100.0
            terminated = True
            # Sound: Level complete
        
        return reward, terminated

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        obs = np.transpose(arr, (1, 0, 2)).astype(np.uint8)
        if self.render_mode == "human":
            self.human_screen.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.FPS)
        return obs

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        self._render_walls()
        self._render_exit()
        self._render_particles()
        self._render_illusion()
        self._render_guards()
        self._render_player()
        self._render_ui()

    def _render_walls(self):
        for r, row in enumerate(self.walls):
            for c, cell in enumerate(row):
                if cell == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, (c * self.TILE_SIZE, r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

    def _render_exit(self):
        pos_x, pos_y = (self.exit_pos + 0.5) * self.TILE_SIZE
        t = self.steps * 0.1
        radius = self.TILE_SIZE * 0.4 + 3 * math.sin(t)
        self._draw_glow_circle(self.screen, (pos_x, pos_y), radius, self.COLOR_EXIT, 15)
    
    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_illusion(self):
        if self.illusion:
            pos = (self.illusion['pos'] + 0.5) * self.TILE_SIZE
            # Main shape
            t = self.steps * 0.2
            radius = self.TILE_SIZE * 0.3
            points = []
            for i in range(5):
                angle = i * (2 * math.pi / 5) + t
                x = pos[0] + radius * math.cos(angle)
                y = pos[1] + radius * math.sin(angle)
                points.append((x, y))
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ILLUSION)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ILLUSION)
            
            # Duration aura
            aura_radius = (self.illusion['duration'] / self.illusion['max_duration']) * self.TILE_SIZE * 0.7
            self._draw_glow_circle(self.screen, pos, aura_radius, self.COLOR_ILLUSION, 10)

    def _render_guards(self):
        for guard in self.guards:
            pos = guard['visual_pos']
            size = self.TILE_SIZE * 0.35
            
            if self.show_vision_cones:
                self._render_vision_cone(guard)

            # Draw guard body with glow
            self._draw_glow_circle(self.screen, pos, size, self.COLOR_GUARD, 10)
            
            # Draw a triangle shape for direction
            angle = guard['facing_angle']
            points = [
                (pos[0] + size * math.cos(angle), pos[1] + size * math.sin(angle)),
                (pos[0] + size * 0.5 * math.cos(angle + 2.5), pos[1] + size * 0.5 * math.sin(angle + 2.5)),
                (pos[0] + size * 0.5 * math.cos(angle - 2.5), pos[1] + size * 0.5 * math.sin(angle - 2.5)),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GUARD)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GUARD)

    def _render_vision_cone(self, guard):
        pos = guard['visual_pos']
        angle = guard['facing_angle']
        cone_length = self.TILE_SIZE * 4
        cone_angle = math.pi / 4

        p1 = pos
        p2 = (pos[0] + cone_length * math.cos(angle - cone_angle / 2),
              pos[1] + cone_length * math.sin(angle - cone_angle / 2))
        p3 = (pos[0] + cone_length * math.cos(angle + cone_angle / 2),
              pos[1] + cone_length * math.sin(angle + cone_angle / 2))
        
        pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), (*self.COLOR_GUARD, 40))
        pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), (*self.COLOR_GUARD, 60))

    def _render_player(self):
        pos = self.player_visual_pos
        radius = self.TILE_SIZE * 0.4
        self._draw_glow_circle(self.screen, pos, radius, self.COLOR_PLAYER, 20)

    def _render_ui(self):
        ui_surface = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_surface.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surface, (0, 0))

        score_text = self.font.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": tuple(self.player_pos.astype(int)),
            "illusion_active": self.illusion is not None,
        }

    # --- HELPER METHODS ---
    def _generate_level(self):
        self.walls = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.walls[0, :] = 1
        self.walls[-1, :] = 1
        self.walls[:, 0] = 1
        self.walls[:, -1] = 1
        
        self.walls[2:8, 4] = 1
        self.walls[2:8, 8] = 1
        self.walls[self.GRID_HEIGHT-8:self.GRID_HEIGHT-2, 6] = 1
        self.walls[4, 9:13] = 1

        self.exit_pos = np.array([self.GRID_WIDTH - 2, self.GRID_HEIGHT // 2])

    def _is_wall(self, x, y):
        if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
            return self.walls[y, x] == 1
        return True

    def _initialize_guards(self):
        guards = []
        
        # Guard 1: Vertical patrol
        path1 = [np.array([2.0, 2.0]), np.array([2.0, self.GRID_HEIGHT - 3.0])]
        guards.append(self._create_guard(0, path1))

        # Guard 2: Horizontal patrol
        path2 = [np.array([5.0, 2.0]), np.array([12.0, 2.0])]
        guards.append(self._create_guard(1, path2))
        
        # Guard 3: Box patrol
        path3 = [
            np.array([10.0, 5.0]), np.array([14.0, 5.0]),
            np.array([14.0, 8.0]), np.array([10.0, 8.0])
        ]
        guards.append(self._create_guard(2, path3))

        return guards

    def _create_guard(self, guard_id, path):
        pos = np.array(path[0], dtype=float)
        return {
            "id": guard_id,
            "pos": pos,
            "visual_pos": pos * self.TILE_SIZE + self.TILE_SIZE / 2,
            "patrol_path": path,
            "patrol_index": 1,
            "is_distracted": False,
            "facing_angle": 0.0,
        }

    def _is_in_vision_cone(self, guard, point):
        cone_length_sq = (self.TILE_SIZE * 4) ** 2
        cone_angle = math.pi / 4

        guard_pos = guard['visual_pos']
        vec_to_point = point - guard_pos
        dist_sq = np.dot(vec_to_point, vec_to_point)

        if dist_sq == 0 or dist_sq > cone_length_sq:
            return False

        guard_facing_vec = np.array([math.cos(guard['facing_angle']), math.sin(guard['facing_angle'])])
        
        # Normalize vec_to_point
        vec_to_point_norm = vec_to_point / np.sqrt(dist_sq)

        dot_product = np.dot(guard_facing_vec, vec_to_point_norm)
        angle_to_point = math.acos(np.clip(dot_product, -1.0, 1.0))

        return angle_to_point < cone_angle / 2

    def _draw_glow_circle(self, surface, pos, radius, color, glow_size):
        pos = (int(pos[0]), int(pos[1]))
        for i in range(glow_size, 0, -1):
            alpha = int(100 * (1 - (i / glow_size))**2)
            current_radius = int(radius + i)
            if current_radius > 0:
                pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], current_radius, (*color, alpha))
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius), color)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    # --- Manual Play Controls ---
    # Arrow keys: Move
    # Space: Place Illusion
    # Shift: Toggle Vision Cones
    # R: Reset
    
    while True:
        # Default action: do nothing
        action = [0, 0, 0] 
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        if keys[pygame.K_r]:
            obs, info = env.reset()
            continue

        if keys[pygame.K_q]:
            break

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()

    env.close()