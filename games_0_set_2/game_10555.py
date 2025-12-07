import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:41:09.647899
# Source Brief: brief_00555.md
# Brief Index: 555
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player guides a sentient seed through a
    procedurally generated forest. The seed can terraform the land by growing
    roots to create new paths, navigating obstacles and hazards to reach
    portals that lead to new areas, with the ultimate goal of reaching the
    ancient tree.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Guide a sentient seed through a forest, growing roots to create paths and reach the ancient tree."
    user_guide = "Controls: Use arrow keys to move. Hold Space to grow a root. Press Shift to rotate your aiming direction."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.TARGET_FPS = 30
        self.FINAL_LEVEL = 5
        self.MAX_STEPS = 1500

        # Colors
        self.COLOR_BG_TOP = (10, 5, 20)
        self.COLOR_BG_BOTTOM = (30, 20, 50)
        self.COLOR_SEED = (255, 255, 100)
        self.COLOR_SEED_GLOW = (255, 255, 100, 50)
        self.COLOR_PLATFORM = (20, 100, 50)
        self.COLOR_PLATFORM_EDGE = (40, 150, 90)
        self.COLOR_ROOT = (139, 69, 19)
        self.COLOR_ROOT_EDGE = (160, 82, 45)
        self.COLOR_PORTAL = [(0, 191, 255), (70, 130, 180), (135, 206, 250)]
        self.COLOR_HAZARD = (255, 0, 50)
        self.COLOR_HAZARD_GLOW = (255, 0, 50, 100)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (0, 255, 127)
        self.COLOR_HEALTH_BAR_BG = (128, 0, 0, 150)
        self.COLOR_AIM_RETICLE = (255, 255, 255, 100)
        self.COLOR_TREE_TRUNK = (83, 53, 10)
        self.COLOR_TREE_LEAVES = (34, 139, 34)

        # Physics & Gameplay
        self.SEED_SPEED = 4.0
        self.SEED_RADIUS = 8
        self.GRAVITY = 0.8
        self.ROOT_GROW_SPEED = 2.0
        self.ROOT_MAX_LENGTH = 100
        self.ROOT_COOLDOWN_MAX = 30 # steps
        self.HAZARD_DAMAGE = 15
        self.FALL_DAMAGE = 100

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 18, bold=True)
        self.font_big = pygame.font.SysFont('Consolas', 48, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.victory = False

        self.seed_pos = pygame.Vector2(0, 0)
        self.seed_velocity = pygame.Vector2(0, 0)
        self.seed_health = 100.0
        self.is_on_ground = False

        self.facing_direction = 1  # 0:up, 1:right, 2:down, 3:left
        self.last_shift_state = False

        self.root_cooldown = 0
        self.is_growing_root = False
        self.current_root_start = None
        self.current_root_end = None

        self.platforms = []
        self.roots = []
        self.portals = []
        self.hazards = []
        self.particles = []
        self.stars = []
        self._generate_stars()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.victory = False

        self.seed_health = 100.0
        self.seed_velocity = pygame.Vector2(0, 0)
        self.is_on_ground = False

        self.facing_direction = 1
        self.last_shift_state = False
        self.root_cooldown = 0
        self.is_growing_root = False
        self.current_root_start = None
        self.current_root_end = None
        
        self.particles.clear()

        self._generate_level()

        start_platform = self.platforms[0]
        self.seed_pos = pygame.Vector2(start_platform.centerx, start_platform.centery)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game State ---
        event_reward = self._update_game_logic()
        reward += event_reward
        reward += 0.01 # Small reward for surviving a step

        # --- Check Termination ---
        self.steps += 1
        terminated = self.seed_health <= 0 or self.steps >= self.MAX_STEPS or self.victory
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            if self.victory:
                # SFX: Victory fanfare
                reward += 100.0
            elif self.seed_health <= 0:
                # SFX: Player death
                reward -= 100.0
            self.game_over = True

        # Clamp reward
        reward = np.clip(reward, -100, 100)
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement and Aiming
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: # Up
            move_vec.y = -1
            self.facing_direction = 0
        elif movement == 2: # Down
            move_vec.y = 1
            self.facing_direction = 2
        elif movement == 3: # Left
            move_vec.x = -1
            self.facing_direction = 3
        elif movement == 4: # Right
            move_vec.x = 1
            self.facing_direction = 1
        
        if move_vec.length() > 0:
            self.seed_velocity = move_vec.normalize() * self.SEED_SPEED
        else:
            self.seed_velocity = pygame.Vector2(0, 0)

        # Rotate Aiming Direction (on press)
        if shift_held and not self.last_shift_state:
            self.facing_direction = (self.facing_direction + 1) % 4
            # SFX: Aim rotate click
        self.last_shift_state = shift_held

        # Grow Root
        if space_held and self.root_cooldown == 0 and self.is_on_ground and not self.is_growing_root:
            self.is_growing_root = True
            self.current_root_start = pygame.Vector2(self.seed_pos)
            self.current_root_end = pygame.Vector2(self.seed_pos)
            # SFX: Root grow start
        
        if not space_held and self.is_growing_root:
            self._finalize_root()

    def _update_game_logic(self):
        event_reward = 0.0

        # Update cooldowns
        if self.root_cooldown > 0:
            self.root_cooldown -= 1

        # Update root growth
        if self.is_growing_root:
            direction_vectors = [pygame.Vector2(0, -1), pygame.Vector2(1, 0), pygame.Vector2(0, 1), pygame.Vector2(-1, 0)]
            direction = direction_vectors[self.facing_direction]
            self.current_root_end += direction * self.ROOT_GROW_SPEED
            if self.current_root_start.distance_to(self.current_root_end) > self.ROOT_MAX_LENGTH:
                self._finalize_root()

        # Update player position
        if self.is_on_ground:
            self.seed_pos += self.seed_velocity
        else: # Apply gravity
            self.seed_velocity.y += self.GRAVITY
            self.seed_pos += self.seed_velocity

        # Boundary checks
        self.seed_pos.x = np.clip(self.seed_pos.x, self.SEED_RADIUS, self.SCREEN_WIDTH - self.SEED_RADIUS)
        if self.seed_pos.y > self.SCREEN_HEIGHT + self.SEED_RADIUS * 2:
            self.seed_health = 0 # Fell off map
            self._create_particles(self.seed_pos, self.COLOR_HAZARD, 20)
        
        # Check ground collision
        self._check_ground_collision()

        # Check hazard collision
        for hazard_pos in self.hazards:
            if self.seed_pos.distance_to(hazard_pos) < self.SEED_RADIUS * 2:
                self.seed_health -= self.HAZARD_DAMAGE
                # SFX: Player hurt
                self._create_particles(self.seed_pos, self.COLOR_HAZARD, 10)
                self.hazards.remove(hazard_pos) # Hazard disappears after hitting
                break
        
        # Check portal collision
        for portal_pos in self.portals:
            if self.seed_pos.distance_to(portal_pos) < 20: # Portal radius
                event_reward += 1.0
                # SFX: Portal entry
                self._create_particles(portal_pos, self.COLOR_PORTAL[0], 50)
                if self.level + 1 == self.FINAL_LEVEL:
                    self.level += 1
                    self._generate_final_level()
                else:
                    self.level += 1
                    self._generate_level()
                
                start_platform = self.platforms[0]
                self.seed_pos = pygame.Vector2(start_platform.centerx, start_platform.centery)
                self.seed_velocity = pygame.Vector2(0, 0)
                break
        
        # Check win condition
        if self.level == self.FINAL_LEVEL and len(self.portals) == 0: # Ancient tree level
            tree_base = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.7)
            if self.seed_pos.distance_to(tree_base) < 40:
                self.victory = True

        self.seed_health = max(0, self.seed_health)

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        return event_reward

    def _finalize_root(self):
        self.is_growing_root = False
        if self.current_root_start.distance_to(self.current_root_end) > self.SEED_RADIUS:
            self.roots.append((self.current_root_start, self.current_root_end))
            # SFX: Root grow complete
            self._create_particles(self.current_root_end, self.COLOR_ROOT, 5)
        self.current_root_start = None
        self.current_root_end = None
        self.root_cooldown = self.ROOT_COOLDOWN_MAX

    def _check_ground_collision(self):
        self.is_on_ground = False
        
        # Check platforms
        for plat in self.platforms:
            if plat.collidepoint(self.seed_pos):
                self.is_on_ground = True
                self.seed_velocity.y = 0
                self.seed_pos.y = plat.top + 1 # Stand on top of platform
                return

        # Check roots
        for start, end in self.roots:
            point = pygame.Vector2(self.seed_pos.x, self.seed_pos.y)
            line_vec = end - start
            line_len_sq = line_vec.length_squared()
            if line_len_sq == 0: continue
            
            t = max(0, min(1, point_vec.dot(line_vec) / line_len_sq))
            closest_point = start + t * line_vec
            
            if point.distance_to(closest_point) < self.SEED_RADIUS:
                self.is_on_ground = True
                self.seed_velocity.y = 0
                self.seed_pos.y = closest_point.y
                return

    def _generate_level(self):
        self.platforms.clear()
        self.roots.clear()
        self.portals.clear()
        self.hazards.clear()

        plat_width, plat_height = 80, 20
        num_platforms = 7 + self.level
        
        # Create starting platform
        start_plat = pygame.Rect(self.SCREEN_WIDTH / 2 - plat_width / 2, self.SCREEN_HEIGHT * 0.8, plat_width, plat_height)
        self.platforms.append(start_plat)
        
        # Procedurally generate connected platforms
        current_platforms = [start_plat]
        for _ in range(num_platforms):
            base_plat = random.choice(current_platforms)
            angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upwards angles
            distance = self.np_random.uniform(80, 120)
            
            new_x = base_plat.centerx + math.cos(angle) * distance
            new_y = base_plat.centery + math.sin(angle) * distance
            
            new_plat = pygame.Rect(new_x - plat_width/2, new_y - plat_height/2, plat_width, plat_height)
            
            # Ensure it's on screen and not overlapping too much
            if 0 < new_plat.left and new_plat.right < self.SCREEN_WIDTH and 0 < new_plat.top:
                self.platforms.append(new_plat)
                current_platforms.append(new_plat)

        # Place portal on the highest platform
        highest_plat = min(self.platforms, key=lambda p: p.top)
        self.portals.append(pygame.Vector2(highest_plat.centerx, highest_plat.top - 20))

        # Place hazards
        hazard_density = 0.05 * self.level
        for plat in self.platforms[1:-1]: # Avoid start and end platforms
             if self.np_random.random() < hazard_density:
                 self.hazards.append(pygame.Vector2(plat.centerx, plat.top))

    def _generate_final_level(self):
        self.platforms.clear()
        self.roots.clear()
        self.portals.clear()
        self.hazards.clear()
        
        plat = pygame.Rect(50, self.SCREEN_HEIGHT * 0.75, self.SCREEN_WIDTH - 100, 40)
        self.platforms.append(plat)

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "health": self.seed_health,
            "victory": self.victory,
        }

    def _render_all(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()

    def _render_background(self):
        # Gradient background
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        
        # Stars
        for star in self.stars:
            pos, radius, alpha = star
            color = (200, 200, 255, alpha)
            pygame.draw.circle(self.screen, color, pos, radius)

    def _render_game_elements(self):
        # Ancient Tree Beacon (distant)
        beacon_x = self.SCREEN_WIDTH / 2
        beacon_y = 60
        if self.level < self.FINAL_LEVEL:
            pygame.gfxdraw.filled_circle(self.screen, int(beacon_x), int(beacon_y), 20, (255, 255, 200, 15))
            pygame.gfxdraw.filled_circle(self.screen, int(beacon_x), int(beacon_y), 10, (255, 255, 200, 30))
            pygame.gfxdraw.filled_circle(self.screen, int(beacon_x), int(beacon_y), 5, (255, 255, 200, 100))

        # Portals
        for pos in self.portals:
            t = self.steps % 60 / 60.0
            for i, color in enumerate(self.COLOR_PORTAL):
                radius = 15 - i * 4
                angle = (t * 2 * math.pi + i * 2 * math.pi / 3)
                px = pos.x + math.cos(angle) * 5
                py = pos.y + math.sin(angle) * 5
                self._draw_aa_circle(self.screen, color, (px, py), radius)

        # Platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_EDGE, plat, 2)
            
        # Roots
        for start, end in self.roots:
            pygame.draw.line(self.screen, self.COLOR_ROOT_EDGE, start, end, 10)
            pygame.draw.line(self.screen, self.COLOR_ROOT, start, end, 6)

        # Growing Root
        if self.is_growing_root:
            pygame.draw.line(self.screen, self.COLOR_ROOT_EDGE, self.current_root_start, self.current_root_end, 10)
            pygame.draw.line(self.screen, self.COLOR_ROOT, self.current_root_start, self.current_root_end, 6)
        
        # Hazards
        for pos in self.hazards:
            points = [(pos.x, pos.y), (pos.x - 8, pos.y - 15), (pos.x + 8, pos.y - 15)]
            self._draw_aa_polygon(self.screen, points, self.COLOR_HAZARD_GLOW)
            self._draw_aa_polygon(self.screen, points, self.COLOR_HAZARD)

        # Ancient Tree
        if self.level == self.FINAL_LEVEL:
            self._render_ancient_tree()
            
        # Particles
        for p in self.particles:
            alpha = p['life'] / p['max_life']
            color = p['color']
            pygame.draw.circle(self.screen, (color[0], color[1], color[2], int(alpha*255)), p['pos'], int(p['radius'] * alpha))

        # Aim Reticle
        if self.is_on_ground:
            direction_vectors = [pygame.Vector2(0, -1), pygame.Vector2(1, 0), pygame.Vector2(0, 1), pygame.Vector2(-1, 0)]
            direction = direction_vectors[self.facing_direction]
            start_pos = self.seed_pos + direction * 15
            end_pos = self.seed_pos + direction * 30
            pygame.draw.line(self.screen, self.COLOR_AIM_RETICLE, start_pos, end_pos, 2)

        # Seed
        glow_surf = pygame.Surface((self.SEED_RADIUS * 4, self.SEED_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_SEED_GLOW, (self.SEED_RADIUS * 2, self.SEED_RADIUS * 2), self.SEED_RADIUS * 2)
        self.screen.blit(glow_surf, (self.seed_pos.x - self.SEED_RADIUS * 2, self.seed_pos.y - self.SEED_RADIUS * 2))
        self._draw_aa_circle(self.screen, self.COLOR_SEED, self.seed_pos, self.SEED_RADIUS)

    def _render_ancient_tree(self):
        trunk_rect = pygame.Rect(self.SCREEN_WIDTH/2 - 15, self.SCREEN_HEIGHT * 0.5, 30, self.SCREEN_HEIGHT * 0.25)
        pygame.draw.rect(self.screen, self.COLOR_TREE_TRUNK, trunk_rect)
        
        canopy_center = (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT * 0.4)
        for i in range(5):
            offset_x = self.np_random.uniform(-40, 40)
            offset_y = self.np_random.uniform(-30, 30)
            radius = self.np_random.uniform(30, 60)
            self._draw_aa_circle(self.screen, self.COLOR_TREE_LEAVES, (canopy_center[0] + offset_x, canopy_center[1] + offset_y), radius)


    def _render_ui(self):
        # Health Bar
        health_bar_width = 150
        health_ratio = self.seed_health / 100.0
        current_health_width = int(health_bar_width * health_ratio)
        
        bg_rect = pygame.Rect(10, 10, health_bar_width, 20)
        health_rect = pygame.Rect(10, 10, current_health_width, 20)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, health_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, bg_rect, 1)

        # Text Info
        level_text = self.font_ui.render(f"Area: {self.level}/{self.FINAL_LEVEL}", True, self.COLOR_UI_TEXT)
        score_text = self.font_ui.render(f"Score: {self.score:.0f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 30))

        # Game Over / Victory Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            
            message = "VICTORY" if self.victory else "TERRAFORMING FAILED"
            color = self.COLOR_SEED if self.victory else self.COLOR_HAZARD
            text_surf = self.font_big.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _generate_stars(self):
        self.stars = []
        for _ in range(100):
            pos = (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT))
            radius = self.np_random.uniform(0.5, 1.5)
            alpha = self.np_random.integers(50, 150)
            self.stars.append((pos, radius, alpha))
            
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _draw_aa_circle(self, surface, color, center, radius):
        x, y, r = int(center[0]), int(center[1]), int(radius)
        if r <= 0: return
        pygame.gfxdraw.aacircle(surface, x, y, r, color)
        pygame.gfxdraw.filled_circle(surface, x, y, r, color)

    def _draw_aa_polygon(self, surface, points, color):
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # The original code had a display here, which is not needed for the fix.
    # To run this in a non-headless mode, you would need to:
    # 1. Remove or comment out: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # 2. Add a display initialization:
    #    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    #    pygame.display.set_caption("Sentient Seed")
    
    terminated = False
    truncated = False
    
    # --- Manual Control Mapping ---
    # Arrows: Move
    # Space: Grow Root
    # Left Shift: Rotate Aim
    
    # This loop is for demonstration and won't run in the test environment
    # because it requires a display and user input handling.
    # It is kept here for reference on how to control the game.
    
    print("Running a test episode with random actions...")
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode finished.")
            obs, info = env.reset()

    print("Test episode finished.")
    env.close()