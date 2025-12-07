
# Generated: 2025-08-28T04:44:32.546707
# Source Brief: brief_02407.md
# Brief Index: 2407

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Use arrow keys (↑↓←→) to apply a push force to all boxes."
    game_description = "Push boxes up treacherous isometric slopes to reach the summit in this timing-based puzzle game."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (30, 35, 40)
    COLOR_SLOPE = (70, 80, 90)
    COLOR_SLOPE_EDGE = (100, 110, 120)
    COLOR_TARGET = (255, 215, 0, 100) # Gold with alpha
    COLOR_TARGET_EDGE = (255, 225, 50)
    BOX_COLORS = [(231, 76, 60), (52, 152, 219), (46, 204, 113), (155, 89, 182)]
    COLOR_SHADOW = (0, 0, 0, 50)
    COLOR_TEXT = (240, 240, 240)
    
    # Physics & World
    GRAVITY = 0.3
    PUSH_FORCE = 0.4
    FRICTION = 0.96
    BOX_SIZE = 20
    SLOPE_WIDTH = 250  # World units
    SLOPE_LENGTH = 450 # World units
    
    # Isometric projection
    ISO_ANGLE = math.pi / 6  # 30 degrees
    ISO_COS = math.cos(ISO_ANGLE)
    ISO_SIN = math.sin(ISO_ANGLE)
    TILE_SCALE = 1.0
    Z_SCALE = 1.0

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
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14)
        
        self.center_x = self.SCREEN_WIDTH // 2
        self.center_y = self.SCREEN_HEIGHT // 2 - 50

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.boxes = []
        self.particles = []
        self.timer = 0
        self.level = 1
        self.max_levels = 3
        self.slope_grade = 0.0
        self.target_zones = []
        self.boxes_in_target = 0
        
        self.reset()
        
        # self.validate_implementation() # Optional self-check

    def _world_to_screen(self, x, y, z):
        sx = self.center_x + (x - y) * self.ISO_COS * self.TILE_SCALE
        sy = self.center_y + (x + y) * self.ISO_SIN * self.TILE_SCALE - z * self.Z_SCALE
        return int(sx), int(sy)

    def _get_slope_z(self, x, y):
        return (x + y) * self.slope_grade

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.game_over and self.boxes_in_target == 4: # Won previous level
            self.level = min(self.max_levels, self.level + 1)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS
        
        # Difficulty scaling
        self.slope_grade = 0.15 + (self.level - 1) * 0.05

        self.boxes = []
        start_x = self.SLOPE_WIDTH / 2
        for i in range(4):
            box_data = {
                "pos": np.array([start_x + (i - 1.5) * (self.BOX_SIZE + 5), 20.0, 0.0], dtype=float),
                "vel": np.array([0.0, 0.0, 0.0], dtype=float),
                "color": self.BOX_COLORS[i],
                "on_target": False,
                "dist_to_target": float('inf')
            }
            box_data["pos"][2] = self._get_slope_z(box_data["pos"][0], box_data["pos"][1])
            self.boxes.append(box_data)

        self.target_zones = []
        target_y = self.SLOPE_LENGTH - self.BOX_SIZE * 2
        for i in range(4):
            self.target_zones.append({
                "pos": np.array([self.SLOPE_WIDTH / 2 + (i - 1.5) * (self.BOX_SIZE + 10), target_y, 0.0]),
                "radius": self.BOX_SIZE * 1.2
            })
            self.target_zones[-1]["pos"][2] = self._get_slope_z(self.target_zones[-1]["pos"][0], self.target_zones[-1]["pos"][1])

        self.particles = []
        self._update_box_distances()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        reward = 0
        self.steps += 1
        self.timer -= 1
        
        total_dist_before = sum(b['dist_to_target'] for b in self.boxes)
        
        # --- Physics Update ---
        self._apply_forces(movement)
        self._update_box_positions()
        self._handle_collisions()
        
        # --- Reward and State Update ---
        self._update_box_distances()
        total_dist_after = sum(b['dist_to_target'] for b in self.boxes)
        
        # Continuous reward for getting closer
        reward += (total_dist_before - total_dist_after) * 0.1

        newly_in_target = self._check_targets()
        reward += newly_in_target * 1.0
        
        fallen_boxes = self._check_fall_off()
        if fallen_boxes > 0:
            reward -= 5.0 * fallen_boxes
            self.score -= 100
            self.game_over = True
        
        self.score += reward

        terminated = self._check_termination()
        if terminated:
            if self.boxes_in_target == 4: # Win
                reward += 100.0
                self.score += 100
            else: # Lose (time out or fell)
                reward -= 100.0
                self.score -= 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _apply_forces(self, movement):
        force = np.array([0.0, 0.0, 0.0])
        if movement == 1: # Up
            force = np.array([self.PUSH_FORCE, self.PUSH_FORCE, 0.0])
        elif movement == 2: # Down
            force = np.array([-self.PUSH_FORCE, -self.PUSH_FORCE, 0.0])
        elif movement == 3: # Left
            force = np.array([-self.PUSH_FORCE, self.PUSH_FORCE, 0.0])
        elif movement == 4: # Right
            force = np.array([self.PUSH_FORCE, -self.PUSH_FORCE, 0.0])
        
        for box in self.boxes:
            box['vel'] += force
            box['vel'][2] -= self.GRAVITY

    def _update_box_positions(self):
        for box in self.boxes:
            box['pos'] += box['vel']
            box['vel'] *= self.FRICTION

    def _handle_collisions(self):
        # Box-Slope collision
        for box in self.boxes:
            slope_z = self._get_slope_z(box['pos'][0], box['pos'][1])
            if box['pos'][2] < slope_z:
                box['pos'][2] = slope_z
                # Bounce effect and friction
                if box['vel'][2] < -0.5:
                    # sound: "thud"
                    self._create_particles(box['pos'], 5, (120, 120, 120))
                box['vel'][2] *= -0.3 # Bounce
                box['vel'][:2] *= 0.9 # Surface friction
        
        # Box-Box collision
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                b1 = self.boxes[i]
                b2 = self.boxes[j]
                
                delta = b1['pos'] - b2['pos']
                dist = np.linalg.norm(delta)
                
                if dist < self.BOX_SIZE:
                    overlap = self.BOX_SIZE - dist
                    direction = delta / (dist + 1e-6)
                    
                    b1['pos'] += direction * overlap / 2
                    b2['pos'] -= direction * overlap / 2
                    
                    # Simple momentum transfer
                    v1 = b1['vel'].copy()
                    v2 = b2['vel'].copy()
                    b1['vel'] = v2
                    b2['vel'] = v1
                    # sound: "clack"
                    self._create_particles(b1['pos'] + delta/2, 3, (180, 180, 180))

    def _update_box_distances(self):
        for box in self.boxes:
            # Find closest target zone (using XY plane distance)
            min_dist = float('inf')
            for target in self.target_zones:
                dist = np.linalg.norm(box['pos'][:2] - target['pos'][:2])
                if dist < min_dist:
                    min_dist = dist
            box['dist_to_target'] = min_dist

    def _check_targets(self):
        newly_in_target = 0
        self.boxes_in_target = 0
        for box in self.boxes:
            is_in_any_target = False
            for target in self.target_zones:
                if np.linalg.norm(box['pos'][:2] - target['pos'][:2]) < target['radius'] / 2:
                    is_in_any_target = True
                    break
            
            if is_in_any_target:
                self.boxes_in_target += 1
                if not box['on_target']:
                    box['on_target'] = True
                    newly_in_target += 1
                    # sound: "chime"
                    self._create_particles(box['pos'], 15, self.COLOR_TARGET[:3])
            else:
                box['on_target'] = False
        return newly_in_target

    def _check_fall_off(self):
        fallen_count = 0
        boxes_to_keep = []
        for box in self.boxes:
            x, y, z = box['pos']
            if 0 <= x < self.SLOPE_WIDTH and 0 <= y < self.SLOPE_LENGTH:
                boxes_to_keep.append(box)
            else:
                fallen_count += 1
                # sound: "whoosh_fall"
                self._create_particles((x,y,z), 30, box['color'])
        
        if fallen_count > 0:
            self.boxes = boxes_to_keep
            self.game_over = True
        return fallen_count

    def _check_termination(self):
        if self.game_over:
            return True
        if self.timer <= 0:
            self.game_over = True
            return True
        if self.boxes_in_target == 4:
            self.game_over = True
            return True
        return False

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
            "timer": self.timer,
            "level": self.level,
            "boxes_in_target": self.boxes_in_target
        }

    def _render_game(self):
        # --- Render Slope and Targets ---
        slope_points = [
            self._world_to_screen(0, 0, self._get_slope_z(0, 0)),
            self._world_to_screen(self.SLOPE_WIDTH, 0, self._get_slope_z(self.SLOPE_WIDTH, 0)),
            self._world_to_screen(self.SLOPE_WIDTH, self.SLOPE_LENGTH, self._get_slope_z(self.SLOPE_WIDTH, self.SLOPE_LENGTH)),
            self._world_to_screen(0, self.SLOPE_LENGTH, self._get_slope_z(0, self.SLOPE_LENGTH))
        ]
        pygame.gfxdraw.filled_polygon(self.screen, slope_points, self.COLOR_SLOPE)
        pygame.gfxdraw.aapolygon(self.screen, slope_points, self.COLOR_SLOPE_EDGE)

        for target in self.target_zones:
            pos = target['pos']
            radius = target['radius']
            center_screen = self._world_to_screen(pos[0], pos[1], pos[2])
            pygame.gfxdraw.filled_circle(self.screen, center_screen[0], center_screen[1], int(radius), self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, center_screen[0], center_screen[1], int(radius), self.COLOR_TARGET_EDGE)
        
        # --- Render Particles ---
        self._update_and_draw_particles()

        # --- Render Boxes and Shadows (sorted by depth) ---
        renderables = self.boxes.copy()
        renderables.sort(key=lambda b: b['pos'][0] + b['pos'][1])

        for box in renderables:
            # Shadow
            shadow_z = self._get_slope_z(box['pos'][0], box['pos'][1])
            shadow_pos = self._world_to_screen(box['pos'][0], box['pos'][1], shadow_z)
            shadow_size = int(self.BOX_SIZE * (1 - (box['pos'][2] - shadow_z) / 200))
            if shadow_size > 0:
                shadow_surf = pygame.Surface((shadow_size*2, shadow_size*2), pygame.SRCALPHA)
                pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, (0,0,shadow_size*2, shadow_size))
                self.screen.blit(shadow_surf, (shadow_pos[0] - shadow_size, shadow_pos[1] - shadow_size/2))

            # Box
            self._draw_iso_cube(box['pos'], box['color'])

    def _draw_iso_cube(self, pos, color):
        x, y, z = pos
        s = self.BOX_SIZE
        
        # 8 vertices of the cube
        points = [
            (x, y, z), (x + s, y, z), (x + s, y + s, z), (x, y + s, z),
            (x, y, z + s), (x + s, y, z + s), (x + s, y + s, z + s), (x, y + s, z + s)
        ]
        screen_points = [self._world_to_screen(px, py, pz) for px, py, pz in points]

        # Darken colors for shading
        color_dark = tuple(max(0, c - 40) for c in color)
        color_darker = tuple(max(0, c - 60) for c in color)
        
        # Draw faces (order matters for 3D illusion)
        # Left face
        pygame.gfxdraw.filled_polygon(self.screen, [screen_points[0], screen_points[3], screen_points[7], screen_points[4]], color_dark)
        # Right face
        pygame.gfxdraw.filled_polygon(self.screen, [screen_points[3], screen_points[2], screen_points[6], screen_points[7]], color_darker)
        # Top face
        pygame.gfxdraw.filled_polygon(self.screen, [screen_points[4], screen_points[5], screen_points[6], screen_points[7]], color)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(1, 4)]),
                "life": random.randint(10, 20),
                "color": color
            })

    def _update_and_draw_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'][2] -= self.GRAVITY * 0.5
            p['life'] -= 1
            
            sx, sy = self._world_to_screen(*p['pos'])
            size = max(1, int(p['life'] / 5))
            pygame.draw.rect(self.screen, p['color'], (sx, sy, size, size))
        
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_ui(self):
        # Timer
        time_text = f"TIME: {self.timer // self.FPS:02d}"
        time_surf = self.font_main.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 20, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))
        
        # Level
        level_text = f"LEVEL: {self.level}/{self.max_levels}"
        level_surf = self.font_small.render(level_text, True, self.COLOR_TEXT)
        self.screen.blit(level_surf, (20, 40))

        # Boxes in target
        box_count_text = f"TARGETS: {self.boxes_in_target}/4"
        box_count_surf = self.font_small.render(box_count_text, True, self.COLOR_TEXT)
        self.screen.blit(box_count_surf, (self.SCREEN_WIDTH - box_count_surf.get_width() - 20, 40))

        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.boxes_in_target == 4:
                msg = f"LEVEL {self.level} COMPLETE!"
            elif self.timer <= 0:
                msg = "TIME UP!"
            else:
                msg = "BOX LOST!"
            
            msg_surf = self.font_main.render(msg, True, (255, 255, 255))
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(msg_surf, msg_rect)

            reset_surf = self.font_small.render("Call reset() to continue", True, (200, 200, 200))
            reset_rect = reset_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20))
            self.screen.blit(reset_surf, reset_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("--- Running Implementation Validation ---")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        print("✓ Action space is correct.")
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        print("✓ Observation space is correct.")
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert "score" in info and "steps" in info
        print("✓ reset() returns correct format.")
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        assert "score" in info and "steps" in info
        print("✓ step() returns correct format.")
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's not part of the required Gymnasium interface but is useful for testing
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy' or 'windows'
    
    env = GameEnv()
    env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Slope Pusher")
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print(f"Game Reset. Level: {info['level']}")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(GameEnv.FPS)

        if terminated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}, Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # In a real game loop, you'd wait for a reset command
            # Here, we'll just keep showing the final state until 'r' is pressed
            pass
            
    env.close()