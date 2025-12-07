import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T19:25:20.633171
# Source Brief: brief_03167.md
# Brief Index: 3167
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Navigate a zero-gravity labyrinth to collect all the runes. "
        "Flip gravity to your advantage and teleport to solve puzzles before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to flip gravity and shift to teleport to the nearest rune."
    )
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    COLOR_BG = (15, 20, 35)
    COLOR_WALL = (70, 80, 100)
    COLOR_PLAYER_NORMAL = (255, 50, 50)
    COLOR_PLAYER_REVERSED = (50, 255, 50)
    COLOR_RUNE_ACTIVE = (255, 215, 0)
    COLOR_RUNE_COLLECTED = (192, 192, 192)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_GRAVITY_ARROW = (150, 150, 180)

    PLAYER_SIZE = 16
    PLAYER_SPEED = 2.5
    PLAYER_FRICTION = 0.85
    GRAVITY_STRENGTH = 0.4
    
    RUNE_SIZE = 10
    
    MAX_STEPS = 2000
    PHYSICS_SUBSTEPS = 4

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.render_mode = render_mode
        self._initialize_state_variables()
        
    def _initialize_state_variables(self):
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 0
        self.world_width = 0
        self.world_height = 0
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.gravity_reversed = False
        self.teleport_charges = 0
        self.runes = []
        self.walls = []
        self.particles = []
        self.camera_pos = np.array([0.0, 0.0])
        self.prev_space_held = 0
        self.prev_shift_held = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.level = getattr(self, 'level', 0) + 1
        
        level_scale_factor = 1 + (0.05 * ((self.level - 1) // 3))
        self.world_width = int(self.SCREEN_WIDTH * 1.5 * level_scale_factor)
        self.world_height = int(self.SCREEN_HEIGHT * 1.5 * level_scale_factor)
        
        num_runes = 3 + (self.level - 1)
        self.teleport_charges = 2 + self.level // 2
        
        self.walls = self._generate_level()
        spawn_pos, self.runes = self._populate_level(num_runes)
        
        self.player_pos = np.array(spawn_pos, dtype=float)
        self.player_vel = np.array([0.0, 0.0])
        self.gravity_reversed = False
        
        self.camera_pos = self.player_pos.copy()
        
        self.particles = []
        self.prev_space_held = 0
        self.prev_shift_held = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        movement, space_held, shift_held = action
        
        dist_before = self._get_dist_to_nearest_rune()
        
        # --- Handle Actions ---
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        if space_pressed:
            self.gravity_reversed = not self.gravity_reversed
            self._create_gravity_flip_effect()
            
        if shift_pressed and self.teleport_charges > 0:
            nearest_rune = self._get_nearest_rune()
            if nearest_rune:
                self._create_teleport_effect(self.player_pos)
                self.player_pos = np.array(nearest_rune['pos'], dtype=float)
                self.player_vel = np.array([0.0, 0.0])
                self._create_teleport_effect(self.player_pos)
                self.teleport_charges -= 1
                reward -= 1.0

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # --- Physics and Movement ---
        for _ in range(self.PHYSICS_SUBSTEPS):
            # Apply gravity
            gravity_acc = self.GRAVITY_STRENGTH * (-1 if self.gravity_reversed else 1)
            self.player_vel[1] += gravity_acc / self.PHYSICS_SUBSTEPS
            
            # Apply movement action
            if movement == 1: self.player_vel[1] -= self.PLAYER_SPEED / self.PHYSICS_SUBSTEPS # Up
            elif movement == 2: self.player_vel[1] += self.PLAYER_SPEED / self.PHYSICS_SUBSTEPS # Down
            elif movement == 3: self.player_vel[0] -= self.PLAYER_SPEED / self.PHYSICS_SUBSTEPS # Left
            elif movement == 4: self.player_vel[0] += self.PLAYER_SPEED / self.PHYSICS_SUBSTEPS # Right
                
            # Apply friction
            self.player_vel *= self.PLAYER_FRICTION ** (1/self.PHYSICS_SUBSTEPS)
            
            # Update position and handle wall collisions
            self.player_pos += self.player_vel
            self._handle_wall_collisions()

        # --- Rune Collection ---
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE/2, self.player_pos[1] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for rune in self.runes:
            if not rune['collected']:
                rune_rect = pygame.Rect(rune['pos'][0] - self.RUNE_SIZE, rune['pos'][1] - self.RUNE_SIZE, self.RUNE_SIZE*2, self.RUNE_SIZE*2)
                if player_rect.colliderect(rune_rect):
                    rune['collected'] = True
                    self.score += 1
                    reward += 5.0
                    self._create_collect_effect(rune['pos'])

        # --- Distance-based Reward ---
        dist_after = self._get_dist_to_nearest_rune()
        if dist_before is not None and dist_after is not None:
            distance_diff = dist_before - dist_after
            if distance_diff > 0.1:
                reward += 0.1 * (distance_diff / self.PLAYER_SPEED)
            elif distance_diff < -0.1:
                reward -= 0.02
        
        # --- Update Particles & Game State ---
        self._update_particles()
        self.steps += 1
        
        # --- Termination ---
        all_collected = all(r['collected'] for r in self.runes)
        max_steps_reached = self.steps >= self.MAX_STEPS
        terminated = all_collected or max_steps_reached
        truncated = False
        
        if all_collected:
            reward += 50.0
        elif max_steps_reached:
            reward -= 10.0 # Penalty for running out of time
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        # Smooth camera
        self.camera_pos = self.camera_pos * 0.9 + self.player_pos * 0.1
        cam_x = self.camera_pos[0] - self.SCREEN_WIDTH / 2
        cam_y = self.camera_pos[1] - self.SCREEN_HEIGHT / 2
        
        self.screen.fill(self.COLOR_BG)
        
        self._render_walls(cam_x, cam_y)
        self._render_runes(cam_x, cam_y)
        self._render_particles(cam_x, cam_y)
        self._render_player(cam_x, cam_y)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "teleport_charges": self.teleport_charges,
            "runes_left": sum(1 for r in self.runes if not r['collected']),
            "level": self.level,
        }

    # --- Rendering Helpers ---
    def _render_walls(self, cam_x, cam_y):
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall.move(-cam_x, -cam_y))

    def _render_runes(self, cam_x, cam_y):
        for rune in self.runes:
            pos = (int(rune['pos'][0] - cam_x), int(rune['pos'][1] - cam_y))
            if rune['collected']:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.RUNE_SIZE, self.COLOR_RUNE_COLLECTED)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.RUNE_SIZE, self.COLOR_RUNE_COLLECTED)
            else:
                # Glow effect
                for i in range(4):
                    alpha = 150 - i * 35
                    radius = self.RUNE_SIZE + i * 2
                    color = (*self.COLOR_RUNE_ACTIVE, alpha)
                    s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(s, color, (radius, radius), radius)
                    self.screen.blit(s, (pos[0]-radius, pos[1]-radius))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.RUNE_SIZE, self.COLOR_RUNE_ACTIVE)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.RUNE_SIZE, self.COLOR_RUNE_ACTIVE)

    def _render_player(self, cam_x, cam_y):
        pos = (int(self.player_pos[0] - cam_x), int(self.player_pos[1] - cam_y))
        color = self.COLOR_PLAYER_REVERSED if self.gravity_reversed else self.COLOR_PLAYER_NORMAL
        
        # Glow effect
        for i in range(4):
            alpha = 100 - i * 25
            size = self.PLAYER_SIZE + i * 4
            glow_color = (*color, alpha)
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=3)
            self.screen.blit(s, (pos[0] - size/2, pos[1] - size/2))

        player_rect = pygame.Rect(pos[0] - self.PLAYER_SIZE/2, pos[1] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, color, player_rect, border_radius=3)

    def _render_particles(self, cam_x, cam_y):
        for p in self.particles:
            pos = (int(p['pos'][0] - cam_x), int(p['pos'][1] - cam_y))
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            radius = int(p['radius'] * (p['life'] / p['max_life']))
            if radius > 0:
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (radius, radius), radius)
                self.screen.blit(s, (pos[0]-radius, pos[1]-radius))

    def _render_ui(self):
        # Rune Counter
        runes_collected = sum(1 for r in self.runes if r['collected'])
        total_runes = len(self.runes)
        rune_text = f"RUNES: {runes_collected}/{total_runes}"
        text_surface = self.font_ui.render(rune_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (10, 10))
        
        # Teleport Counter
        teleport_text = f"TELEPORT: {self.teleport_charges}"
        text_surface = self.font_ui.render(teleport_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - text_surface.get_width() - 10, 10))

        # Gravity Indicator
        arrow_points = [(self.SCREEN_WIDTH - 30, self.SCREEN_HEIGHT - 40), (self.SCREEN_WIDTH - 20, self.SCREEN_HEIGHT - 20), (self.SCREEN_WIDTH - 10, self.SCREEN_HEIGHT - 40)]
        if not self.gravity_reversed:
             arrow_points = [(p[0], self.SCREEN_HEIGHT - (p[1] - (self.SCREEN_HEIGHT - 40)) - 20) for p in arrow_points]
        pygame.draw.polygon(self.screen, self.COLOR_GRAVITY_ARROW, arrow_points)

    # --- Logic Helpers ---
    def _handle_wall_collisions(self):
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE/2, self.player_pos[1] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        for wall in self.walls:
            if player_rect.colliderect(wall):
                # Horizontal collision
                if self.player_vel[0] > 0 and player_rect.right > wall.left:
                    self.player_pos[0] = wall.left - self.PLAYER_SIZE/2
                    self.player_vel[0] = 0
                elif self.player_vel[0] < 0 and player_rect.left < wall.right:
                    self.player_pos[0] = wall.right + self.PLAYER_SIZE/2
                    self.player_vel[0] = 0
                
                # Update rect for vertical check
                player_rect.x = self.player_pos[0] - self.PLAYER_SIZE/2
                
                # Vertical collision
                if self.player_vel[1] > 0 and player_rect.bottom > wall.top:
                    self.player_pos[1] = wall.top - self.PLAYER_SIZE/2
                    self.player_vel[1] = 0
                elif self.player_vel[1] < 0 and player_rect.top < wall.bottom:
                    self.player_pos[1] = wall.bottom + self.PLAYER_SIZE/2
                    self.player_vel[1] = 0

    def _get_dist_to_nearest_rune(self):
        uncollected_runes = [r for r in self.runes if not r['collected']]
        if not uncollected_runes:
            return None
        
        distances = [np.linalg.norm(self.player_pos - np.array(r['pos'])) for r in uncollected_runes]
        return min(distances)

    def _get_nearest_rune(self):
        uncollected_runes = [r for r in self.runes if not r['collected']]
        if not uncollected_runes:
            return None
        
        return min(uncollected_runes, key=lambda r: np.linalg.norm(self.player_pos - np.array(r['pos'])))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    # --- Effect Creation ---
    def _create_teleport_effect(self, pos):
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'radius': random.uniform(3, 8),
                'color': random.choice([self.COLOR_PLAYER_NORMAL, self.COLOR_PLAYER_REVERSED, (200, 200, 255)]),
                'life': random.randint(15, 30),
                'max_life': 30
            })
            
    def _create_collect_effect(self, pos):
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'radius': random.uniform(2, 5),
                'color': self.COLOR_RUNE_ACTIVE,
                'life': random.randint(10, 20),
                'max_life': 20
            })

    def _create_gravity_flip_effect(self):
        for i in range(self.SCREEN_WIDTH // 20):
            self.particles.append({
                'pos': np.array([i * 20 + 10, self.SCREEN_HEIGHT if self.gravity_reversed else 0], dtype=float),
                'vel': np.array([0, -2 if self.gravity_reversed else 2]),
                'radius': 5,
                'color': self.COLOR_GRAVITY_ARROW,
                'life': self.SCREEN_HEIGHT // 2,
                'max_life': self.SCREEN_HEIGHT // 2
            })
            
    # --- Level Generation ---
    def _generate_level(self):
        walls = []
        # Outer bounds
        walls.append(pygame.Rect(0, 0, self.world_width, 10))
        walls.append(pygame.Rect(0, self.world_height - 10, self.world_width, 10))
        walls.append(pygame.Rect(0, 0, 10, self.world_height))
        walls.append(pygame.Rect(self.world_width - 10, 0, 10, self.world_height))

        # Simple random rooms
        num_rooms = 5 + self.level
        for _ in range(num_rooms):
            w = self.np_random.integers(50, 200)
            h = self.np_random.integers(50, 200)
            x = self.np_random.integers(10, self.world_width - w - 10)
            y = self.np_random.integers(10, self.world_height - h - 10)
            walls.append(pygame.Rect(x, y, w, h))
        return walls

    def _is_valid_spawn(self, pos, radius, walls):
        rect = pygame.Rect(pos[0] - radius, pos[1] - radius, radius*2, radius*2)
        if not (0 < rect.left and rect.right < self.world_width and 0 < rect.top and rect.bottom < self.world_height):
            return False
        return not any(rect.colliderect(wall) for wall in walls)

    def _populate_level(self, num_runes):
        spawn_pos = None
        while not spawn_pos:
            pos = (self.np_random.uniform(50, self.world_width-50), self.np_random.uniform(50, self.world_height-50))
            if self._is_valid_spawn(pos, self.PLAYER_SIZE, self.walls):
                spawn_pos = pos

        runes = []
        while len(runes) < num_runes:
            pos = (self.np_random.uniform(50, self.world_width-50), self.np_random.uniform(50, self.world_height-50))
            if self._is_valid_spawn(pos, self.RUNE_SIZE, self.walls):
                # Ensure not too close to other runes
                if all(np.linalg.norm(np.array(pos) - np.array(r['pos'])) > 50 for r in runes):
                    runes.append({'pos': pos, 'collected': False})
        
        return spawn_pos, runes

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you might need to comment out the line: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # at the top of the file, as it requires a display.
    
    # Re-enable display for manual testing if it was disabled
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
    
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("GameEnv Test")
    
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Main game loop
    while not terminated:
        movement_action = 0 # None
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                break
        if terminated:
            continue
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4

        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        obs, reward, term, trunc, info = env.step(action)
        terminated = term or trunc
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.metadata["render_fps"])

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # To play again, uncomment the following two lines:
            # obs, info = env.reset()
            # terminated = False

    env.close()
    pygame.quit()