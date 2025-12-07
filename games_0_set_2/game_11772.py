import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

# Helper classes defined outside the main environment class for clarity

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, vx, vy, radius, color, lifetime):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(vx, vy)
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.initial_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.radius = max(0, self.radius * (self.lifetime / self.initial_lifetime))

    def draw(self, surface, camera_offset):
        if self.lifetime > 0:
            pos_on_screen = self.pos - camera_offset
            pygame.gfxdraw.filled_circle(
                surface, int(pos_on_screen.x), int(pos_on_screen.y), int(self.radius), self.color
            )

class FrostElemental:
    """An enemy that patrols and chases the player."""
    def __init__(self, patrol_points, speed):
        self.patrol_points = [pygame.Vector2(p) for p in patrol_points]
        self.pos = self.patrol_points[0].copy()
        self.speed = speed
        self.state = "PATROL"
        self.patrol_target_idx = 1
        self.vision_radius = 150
        self.attack_radius = 20
        self.was_chasing = False

    def update(self, player_pos, is_player_stealthed):
        self.was_chasing = self.state == "CHASE"
        distance_to_player = self.pos.distance_to(player_pos)

        if not is_player_stealthed and distance_to_player < self.vision_radius:
            self.state = "CHASE"
        elif is_player_stealthed or distance_to_player > self.vision_radius * 1.2:
            self.state = "PATROL"

        if self.state == "CHASE":
            direction = (player_pos - self.pos).normalize()
            self.pos += direction * self.speed * 1.5 # Chases faster
        else: # PATROL
            target = self.patrol_points[self.patrol_target_idx]
            if self.pos.distance_to(target) < self.speed:
                self.patrol_target_idx = (self.patrol_target_idx + 1) % len(self.patrol_points)
                target = self.patrol_points[self.patrol_target_idx]
            direction = (target - self.pos).normalize()
            self.pos += direction * self.speed
            
    def draw(self, surface, camera_offset):
        pos_on_screen = self.pos - camera_offset
        # Body
        pygame.draw.circle(surface, (150, 50, 200), pos_on_screen, 15)
        # Glow
        pygame.gfxdraw.filled_circle(surface, int(pos_on_screen.x), int(pos_on_screen.y), 20, (180, 80, 230, 50))
        # Eyes
        eye_offset = pygame.Vector2(5, -4)
        pygame.draw.circle(surface, (255, 255, 255), pos_on_screen + eye_offset, 3)
        pygame.draw.circle(surface, (255, 255, 255), pos_on_screen - eye_offset, 3)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a frozen cavern, matching pairs of ancient runes to thaw a path to the exit. "
        "Use your blizzard ability to evade patrolling frost elementals."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press Shift to activate a blizzard for stealth. "
        "Fill the combo meter by matching runes and press Space to unleash a thaw explosion."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Colors
    COLOR_BG = (15, 25, 40)
    COLOR_ICE_FROZEN = (60, 80, 120)
    COLOR_ICE_THAWED = (40, 60, 90)
    COLOR_PLAYER = (255, 120, 0)
    COLOR_PLAYER_GLOW = (255, 180, 50, 40)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_COMBO_BAR_BG = (50, 50, 80)
    COLOR_COMBO_BAR_FILL = (0, 200, 255)
    RUNE_COLORS = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 100)]
    
    # Game parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WORLD_WIDTH, WORLD_HEIGHT = 1280, 800
    PLAYER_SPEED = 4.0
    PLAYER_RADIUS = 12
    PLAYER_MAX_HEALTH = 3
    PLAYER_MATCH_RADIUS = 80
    COMBO_METER_MAX = 100
    BLIZZARD_DURATION = 150 # 5 seconds at 30 FPS
    BLIZZARD_COOLDOWN = 300 # 10 seconds
    MAX_STEPS = 2000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_health = None
        self.elementals = []
        self.runes = []
        self.ice_grid = None
        self.exit_rect = None
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.combo_meter = 0
        
        self.blizzard_timer = 0
        self.blizzard_cooldown_timer = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.camera_offset = pygame.Vector2(0, 0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(100, self.WORLD_HEIGHT / 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        
        self.combo_meter = 0
        self.blizzard_timer = 0
        self.blizzard_cooldown_timer = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.particles.clear()
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Generate ice grid
        self.ice_grid = np.ones((self.WORLD_WIDTH // 32, self.WORLD_HEIGHT // 32), dtype=int)
        # Carve a path
        path_y = self.ice_grid.shape[1] // 2
        for x in range(self.ice_grid.shape[0]):
            for y_offset in range(-1, 2):
                if 0 <= path_y + y_offset < self.ice_grid.shape[1]:
                    self.ice_grid[x, path_y + y_offset] = 0
            if self.np_random.random() < 0.2 and x > 5 and x < self.ice_grid.shape[0] - 5:
                path_y += self.np_random.choice([-1, 1])
                path_y = np.clip(path_y, 1, self.ice_grid.shape[1] - 2)

        # Place runes in pairs
        self.runes = []
        num_rune_pairs = 6
        for i in range(num_rune_pairs):
            color_idx = i % len(self.RUNE_COLORS)
            # Find two valid, separated spots
            p1 = self._get_random_frozen_pos()
            p2 = self._get_random_frozen_pos(min_dist_from=p1, min_dist=200)
            self.runes.append({"pos": p1, "type": color_idx, "active": True})
            self.runes.append({"pos": p2, "type": color_idx, "active": True})

        # Place elementals
        self.elementals = []
        for _ in range(3):
            p1 = self._get_random_open_pos()
            p2 = self._get_random_open_pos(min_dist_from=p1, min_dist=150)
            speed = 1.0 + self.np_random.uniform(0.0, 0.5)
            self.elementals.append(FrostElemental([p1, p2], speed))
            
        # Place exit
        self.exit_rect = pygame.Rect(self.WORLD_WIDTH - 80, self.WORLD_HEIGHT / 2 - 40, 80, 80)

    def _get_random_frozen_pos(self, min_dist_from=None, min_dist=0):
        while True:
            x = self.np_random.integers(1, self.ice_grid.shape[0] - 1) * 32 + 16
            y = self.np_random.integers(1, self.ice_grid.shape[1] - 1) * 32 + 16
            if self.ice_grid[x // 32, y // 32] == 1:
                pos = pygame.Vector2(x, y)
                if min_dist_from is None or pos.distance_to(min_dist_from) > min_dist:
                    return pos

    def _get_random_open_pos(self, min_dist_from=None, min_dist=0):
        while True:
            x = self.np_random.integers(1, self.ice_grid.shape[0] - 1) * 32 + 16
            y = self.np_random.integers(1, self.ice_grid.shape[1] - 1) * 32 + 16
            if self.ice_grid[x // 32, y // 32] == 0:
                pos = pygame.Vector2(x, y)
                if min_dist_from is None or pos.distance_to(min_dist_from) > min_dist:
                    return pos

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        # --- Handle Input and State Updates ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        # Update timers
        if self.blizzard_timer > 0: self.blizzard_timer -= 1
        if self.blizzard_cooldown_timer > 0: self.blizzard_cooldown_timer -= 1
        is_stealthed = self.blizzard_timer > 0
        
        # Player movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            new_pos = self.player_pos + move_vec * self.PLAYER_SPEED
            # World boundary collision
            new_pos.x = np.clip(new_pos.x, self.PLAYER_RADIUS, self.WORLD_WIDTH - self.PLAYER_RADIUS)
            new_pos.y = np.clip(new_pos.y, self.PLAYER_RADIUS, self.WORLD_HEIGHT - self.PLAYER_RADIUS)
            # Ice wall collision
            grid_x, grid_y = int(new_pos.x / 32), int(new_pos.y / 32)
            if self.ice_grid[grid_x, grid_y] == 0:
                self.player_pos = new_pos

        # Activate abilities
        if shift_pressed and self.blizzard_cooldown_timer == 0:
            self.blizzard_timer = self.BLIZZARD_DURATION
            self.blizzard_cooldown_timer = self.BLIZZARD_COOLDOWN
            # sfx: Blizzard cast
            for _ in range(100):
                self.particles.append(Particle(
                    self.player_pos.x + self.np_random.uniform(-self.SCREEN_WIDTH/2, self.SCREEN_WIDTH/2),
                    self.player_pos.y + self.np_random.uniform(-self.SCREEN_HEIGHT/2, self.SCREEN_HEIGHT/2),
                    self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1),
                    self.np_random.uniform(2, 5), (255, 255, 255, 150), 60
                ))
        
        if space_pressed and self.combo_meter >= self.COMBO_METER_MAX:
            reward += 1.0 # Event reward for using combo
            self.combo_meter = 0
            # sfx: Combo thaw explosion
            self._thaw_area(self.player_pos, 120)
            for i in range(360):
                if i % 5 == 0:
                    angle = math.radians(i)
                    vx = math.cos(angle) * 5
                    vy = math.sin(angle) * 5
                    self.particles.append(Particle(self.player_pos.x, self.player_pos.y, vx, vy, 8, self.COLOR_PLAYER, 30))

        # --- Game Logic ---
        # Rune matching
        nearby_runes = [r for r in self.runes if r["active"] and self.player_pos.distance_to(pygame.Vector2(r["pos"])) < self.PLAYER_MATCH_RADIUS]
        matched_indices = set()
        for i in range(len(nearby_runes)):
            for j in range(i + 1, len(nearby_runes)):
                r1 = nearby_runes[i]
                r2 = nearby_runes[j]
                if r1["type"] == r2["type"] and i not in matched_indices and j not in matched_indices:
                    r1["active"] = False
                    r2["active"] = False
                    matched_indices.add(i)
                    matched_indices.add(j)
                    reward += 0.1 # Continuous reward for matching
                    self.score += 10
                    self.combo_meter = min(self.COMBO_METER_MAX, self.combo_meter + 25)
                    self._thaw_path(r1["pos"], r2["pos"])
                    # sfx: Rune match success

        # Update elementals
        for e in self.elementals:
            e.update(self.player_pos, is_stealthed)
            # Reward for evading a chasing elemental
            if e.was_chasing and is_stealthed:
                reward += 5.0
            # Penalty for being near an elemental
            if not is_stealthed and e.pos.distance_to(self.player_pos) < e.vision_radius:
                reward -= 0.1
            # Collision with elemental
            if e.pos.distance_to(self.player_pos) < e.attack_radius + self.PLAYER_RADIUS:
                self.player_health -= 1
                self.player_pos = pygame.Vector2(100, self.WORLD_HEIGHT / 2) # Respawn at start
                # sfx: Player takes damage
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            for e in self.elementals:
                e.speed += 0.05
                
        # Update particles
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

        # --- Termination Checks ---
        terminated = False
        truncated = False
        if self.player_health <= 0:
            reward = -100.0
            terminated = True
            self.game_over = True
            # sfx: Game over
            
        if self.exit_rect.collidepoint(self.player_pos):
            reward = 100.0
            terminated = True
            self.game_over = True
            self.score += 1000
            # sfx: Level complete
            
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _thaw_path(self, p1, p2):
        # Bresenham's line algorithm to thaw a path
        x1, y1 = int(p1.x / 32), int(p1.y / 32)
        x2, y2 = int(p2.x / 32), int(p2.y / 32)
        dx, dy = abs(x2 - x1), -abs(y2 - y1)
        sx, sy = 1 if x1 < x2 else -1, 1 if y1 < y2 else -1
        err = dx + dy
        while True:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    self._thaw_tile(x1 + i, y1 + j)
            if x1 == x2 and y1 == y2: break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

    def _thaw_area(self, center, radius):
        cx, cy = int(center.x / 32), int(center.y / 32)
        r_grid = int(radius / 32)
        for x in range(cx - r_grid, cx + r_grid + 1):
            for y in range(cy - r_grid, cy + r_grid + 1):
                if (x-cx)**2 + (y-cy)**2 <= r_grid**2:
                    self._thaw_tile(x, y)

    def _thaw_tile(self, x, y):
        if 0 <= x < self.ice_grid.shape[0] and 0 <= y < self.ice_grid.shape[1]:
            self.ice_grid[x, y] = 0

    def _get_observation(self):
        # Update camera to follow player
        self.camera_offset.x = self.player_pos.x - self.SCREEN_WIDTH / 2
        self.camera_offset.y = self.player_pos.y - self.SCREEN_HEIGHT / 2
        # Clamp camera to world bounds
        self.camera_offset.x = np.clip(self.camera_offset.x, 0, self.WORLD_WIDTH - self.SCREEN_WIDTH)
        self.camera_offset.y = np.clip(self.camera_offset.y, 0, self.WORLD_HEIGHT - self.SCREEN_HEIGHT)

        # --- Render ---
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw ice grid
        start_x = int(self.camera_offset.x // 32)
        end_x = int((self.camera_offset.x + self.SCREEN_WIDTH) // 32) + 1
        start_y = int(self.camera_offset.y // 32)
        end_y = int((self.camera_offset.y + self.SCREEN_HEIGHT) // 32) + 1
        
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                if 0 <= x < self.ice_grid.shape[0] and 0 <= y < self.ice_grid.shape[1]:
                    color = self.COLOR_ICE_FROZEN if self.ice_grid[x, y] == 1 else self.COLOR_ICE_THAWED
                    rect = pygame.Rect(x * 32 - self.camera_offset.x, y * 32 - self.camera_offset.y, 32, 32)
                    pygame.draw.rect(self.screen, color, rect)

        # Draw exit
        exit_screen_rect = self.exit_rect.move(-self.camera_offset.x, -self.camera_offset.y)
        pygame.draw.rect(self.screen, (200, 255, 200), exit_screen_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), exit_screen_rect, 2)

        # Draw runes
        for rune in self.runes:
            if rune["active"]:
                pos_on_screen = pygame.Vector2(rune["pos"]) - self.camera_offset
                color = self.RUNE_COLORS[rune["type"]]
                pygame.gfxdraw.filled_circle(self.screen, int(pos_on_screen.x), int(pos_on_screen.y), 10, color)
                pygame.gfxdraw.filled_circle(self.screen, int(pos_on_screen.x), int(pos_on_screen.y), 13, (*color, 50))
        
        # Draw elementals
        for e in self.elementals:
            e.draw(self.screen, self.camera_offset)

        # Draw player
        player_screen_pos = self.player_pos - self.camera_offset
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, int(player_screen_pos.x), int(player_screen_pos.y), self.PLAYER_RADIUS + 5, self.COLOR_PLAYER_GLOW)
        # Body
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, player_screen_pos, self.PLAYER_RADIUS)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen, self.camera_offset)
            
        # Blizzard overlay
        if self.blizzard_timer > 0:
            overlay = self.screen.copy()
            overlay.set_alpha(80)
            overlay.fill((200, 220, 255))
            self.screen.blit(overlay, (0, 0))

    def _render_ui(self):
        # Health
        for i in range(self.PLAYER_MAX_HEALTH):
            pos = (20 + i * 35, 25)
            color = (150, 200, 255) if i < self.player_health else (50, 60, 80)
            points = [(pos[0], pos[1]-10), (pos[0]+10, pos[1]), (pos[0], pos[1]+10), (pos[0]-10, pos[1])]
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, self.COLOR_UI_TEXT, points, 1)
        
        # Combo Meter
        bar_rect_bg = pygame.Rect(20, 50, 150, 15)
        pygame.draw.rect(self.screen, self.COLOR_COMBO_BAR_BG, bar_rect_bg, border_radius=3)
        fill_width = (self.combo_meter / self.COMBO_METER_MAX) * 150
        bar_rect_fill = pygame.Rect(20, 50, fill_width, 15)
        pygame.draw.rect(self.screen, self.COLOR_COMBO_BAR_FILL, bar_rect_fill, border_radius=3)
        
        # Blizzard Cooldown
        if self.blizzard_cooldown_timer > 0:
            cooldown_prop = self.blizzard_cooldown_timer / self.BLIZZARD_COOLDOWN
            color = (255, 100, 100) if self.blizzard_timer == 0 else (100, 100, 255)
            text = "RDY" if self.blizzard_timer == 0 else f"{self.blizzard_timer/30:.1f}"
            
            s = 40
            pygame.draw.rect(self.screen, self.COLOR_COMBO_BAR_BG, (self.SCREEN_WIDTH - s - 15, 15, s, s), border_radius=5)
            if cooldown_prop < 1.0:
                pygame.draw.arc(self.screen, color, (self.SCREEN_WIDTH - s - 15, 15, s, s), -math.pi/2, -math.pi/2 + (1-cooldown_prop) * 2 * math.pi, 5)
            
            shift_text = self.font_small.render("SHIFT", True, self.COLOR_UI_TEXT)
            self.screen.blit(shift_text, (self.SCREEN_WIDTH - s, 28))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, self.SCREEN_HEIGHT - 30))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            msg = "VICTORY!" if self.player_health > 0 and self.steps < self.MAX_STEPS else "DEFEATED"
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "combo_meter": self.combo_meter,
            "blizzard_cooldown": self.blizzard_cooldown_timer
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Make sure to remove the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Glacial Cavern")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not terminated and not truncated:
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        obs, reward, term, trunc, info = env.step(action)
        terminated = term
        truncated = trunc
        total_reward += reward
        
        # The environment's observation is a numpy array. For display, convert it back.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            terminated = False
            truncated = False

        clock.tick(30) # Run at 30 FPS

    env.close()