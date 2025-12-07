import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:54:41.565740
# Source Brief: brief_02325.md
# Brief Index: 2325
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game entities
class Particle:
    """A single particle for the gas cloud effect."""
    def __init__(self, x, y, color):
        self.pos = pygame.Vector2(x, y)
        self.angle = random.uniform(0, 2 * math.pi)
        self.distance_from_center = random.uniform(5, 20)
        self.rotation_speed = random.uniform(0.01, 0.03) * random.choice([-1, 1])
        self.color = color
        self.lifespan = random.randint(60, 120) # in frames
        self.initial_lifespan = self.lifespan
        self.radius = random.uniform(1, 3)

    def update(self, center_pos):
        self.lifespan -= 1
        self.angle += self.rotation_speed
        self.pos.x = center_pos.x + math.cos(self.angle) * self.distance_from_center
        self.pos.y = center_pos.y + math.sin(self.angle) * self.distance_from_center
        self.distance_from_center *= 0.99 # Slowly spiral inwards

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.initial_lifespan))
            # Use a lighter version of the base color for a glowing effect
            glow_color = tuple(min(255, c + 50) for c in self.color)
            
            # Create a temporary surface for alpha blending
            particle_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, glow_color + (int(alpha * 0.5),), (self.radius, self.radius), self.radius)
            surface.blit(particle_surf, (int(self.pos.x - self.radius), int(self.pos.y - self.radius)))


class Cloud:
    """Represents a color-coded gas cloud."""
    def __init__(self, x, y, color, radius, cloud_id):
        self.id = cloud_id
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.color = color
        self.radius = radius
        self.mass = radius**2
        self.particles = [Particle(self.pos.x, self.pos.y, self.color) for _ in range(int(self.radius * 2))]
        self.target_particle_count = int(self.radius * 2)

    def update(self, screen_width, screen_height):
        # Apply friction/drag
        self.vel *= 0.95
        self.pos += self.vel

        # Boundary checks
        self.pos.x = np.clip(self.pos.x, self.radius, screen_width - self.radius)
        self.pos.y = np.clip(self.pos.y, self.radius, screen_height - self.radius)

        # Update particles
        for p in self.particles:
            p.update(self.pos)
        self.particles = [p for p in self.particles if p.lifespan > 0]
        
        # Replenish particles
        while len(self.particles) < self.target_particle_count:
            self.particles.append(Particle(self.pos.x, self.pos.y, self.color))

    def draw(self, surface):
        # Draw particles first for the cloud body
        for p in self.particles:
            p.draw(surface)
            
        # Draw a soft core glow
        core_radius = int(self.radius * 0.7)
        glow_surf = pygame.Surface((core_radius * 2, core_radius * 2), pygame.SRCALPHA)
        glow_color = self.color + (50,) # Low alpha
        pygame.draw.circle(glow_surf, glow_color, (core_radius, core_radius), core_radius)
        surface.blit(glow_surf, (int(self.pos.x - core_radius), int(self.pos.y - core_radius)))

    def apply_force(self, force):
        self.vel += force / self.mass

    def merge(self, other_cloud):
        # sfx: merge_clouds
        total_mass = self.mass + other_cloud.mass
        self.pos = (self.pos * self.mass + other_cloud.pos * other_cloud.mass) / total_mass
        self.vel = (self.vel * self.mass + other_cloud.vel * other_cloud.mass) / total_mass
        self.radius = math.sqrt(self.radius**2 + other_cloud.radius**2)
        self.mass = total_mass
        self.target_particle_count = int(self.radius * 2)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A cosmic puzzle game. Pick up and merge gaseous clouds to recreate specific celestial patterns against the clock."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Hold space to pick up a gas cloud and release to drop it."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 28, bold=True)
        
        # --- Color Palette ---
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 255)
        self.GAS_COLORS = {
            "RED": (255, 80, 80),
            "BLUE": (80, 120, 255),
            "GREEN": (80, 255, 80),
            "YELLOW": (255, 255, 80),
            "MAGENTA": (255, 80, 255),
        }
        self.COLOR_NAMES = list(self.GAS_COLORS.keys())
        self.unlocked_colors = ["RED", "BLUE"]

        # --- Game Constants ---
        self.MAX_EPISODE_STEPS = 1000
        self.LEVEL_TIME_LIMIT = 60 * 30 # 60 seconds at 30 FPS
        self.CURSOR_SPEED = 8
        self.PHYSICS_FORCE_MULTIPLIER = 0.05
        self.MERGE_DISTANCE_FACTOR = 0.8
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.level_timer = 0
        self.clouds = []
        self.target_pattern = []
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.selected_cloud = None
        self.prev_space_held = False
        self.next_cloud_id = 0
        self._generate_starfield()

        # Initialize state variables for the first time
        # self.reset() # This is called by the wrapper/runner, not needed here.
        
        # --- Final Validation ---
        # self.validate_implementation() # No need to run this in constructor

    def _generate_starfield(self):
        self.stars = []
        for _ in range(200):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            size = random.choice([1, 1, 1, 2])
            brightness = random.randint(50, 150)
            self.stars.append(((x, y), size, (brightness, brightness, brightness)))

    def _start_new_level(self):
        self.level_timer = self.LEVEL_TIME_LIMIT
        self.selected_cloud = None
        
        # Unlock new colors based on level
        if self.level == 3 and "GREEN" not in self.unlocked_colors:
            self.unlocked_colors.append("GREEN")
        if self.level == 5 and "YELLOW" not in self.unlocked_colors:
            self.unlocked_colors.append("YELLOW")
        if self.level == 8 and "MAGENTA" not in self.unlocked_colors:
            self.unlocked_colors.append("MAGENTA")

        self._generate_target_pattern()
        self._spawn_clouds()

    def _generate_target_pattern(self):
        self.target_pattern = []
        num_targets = min(len(self.unlocked_colors), 1 + self.level // 2)
        
        available_positions = []
        for i in range(num_targets):
            while True:
                pos = pygame.Vector2(random.randint(100, self.WIDTH - 100), random.randint(100, self.HEIGHT - 100))
                if all(pos.distance_to(p['pos']) > 150 for p in available_positions):
                    available_positions.append({'pos': pos})
                    break

        colors_for_level = random.sample(self.unlocked_colors, num_targets)
        for i in range(num_targets):
            self.target_pattern.append({
                "pos": available_positions[i]['pos'],
                "color_name": colors_for_level[i],
                "radius": 50,
                "completed": False,
            })

    def _spawn_clouds(self):
        self.clouds = []
        self.next_cloud_id = 0
        num_clouds_per_color = 2 + self.level // 4
        
        for target in self.target_pattern:
            for _ in range(num_clouds_per_color):
                while True:
                    pos = pygame.Vector2(random.uniform(50, self.WIDTH - 50), random.uniform(50, self.HEIGHT - 50))
                    # Ensure new clouds don't spawn inside a target region
                    if not any(pos.distance_to(t['pos']) < t['radius'] for t in self.target_pattern):
                        break
                
                cloud_color = self.GAS_COLORS[target["color_name"]]
                self.clouds.append(Cloud(pos.x, pos.y, cloud_color, random.uniform(15, 20), self.next_cloud_id))
                self.next_cloud_id += 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.unlocked_colors = ["RED", "BLUE"]
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.prev_space_held = False
        
        self._start_new_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- Handle Input and Update Cursor ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        space_pressed = space_held and not self.prev_space_held
        space_released = not space_held and self.prev_space_held

        if space_pressed:
            for cloud in sorted(self.clouds, key=lambda c: c.pos.distance_to(self.cursor_pos)):
                if cloud.pos.distance_to(self.cursor_pos) < cloud.radius:
                    self.selected_cloud = cloud
                    # sfx: pickup
                    break
        
        if space_released and self.selected_cloud:
            # sfx: drop
            self.selected_cloud = None
        
        self.prev_space_held = space_held

        # --- Pre-physics state for reward calculation ---
        old_distances = self._get_cloud_target_distances()

        # --- Update Game Logic ---
        if self.selected_cloud:
            self.selected_cloud.pos.update(self.cursor_pos)
            self.selected_cloud.vel = pygame.Vector2(0, 0)
        
        self._update_physics()
        merge_reward = self._handle_merges()
        reward += merge_reward

        for cloud in self.clouds:
            if cloud != self.selected_cloud:
                cloud.update(self.WIDTH, self.HEIGHT)

        # --- Continuous Reward Calculation ---
        new_distances = self._get_cloud_target_distances()
        for cloud_id, old_dist in old_distances.items():
            if cloud_id in new_distances:
                dist_change = old_dist - new_distances[cloud_id]
                reward += dist_change * 0.001 # Small reward for getting closer

        # --- Level Timer and End Conditions ---
        self.level_timer -= 1
        win_reward, all_complete = self._check_win_condition()
        reward += win_reward

        if all_complete:
            # sfx: level_complete
            self.score += 50
            reward += 50
            self.level += 1
            self._start_new_level()
        elif self.level_timer <= 0:
            # sfx: level_fail
            self.score -= 20
            reward -= 20
            self.level += 1 # Move to next level even on failure
            self._start_new_level()

        self.steps += 1
        terminated = self.steps >= self.MAX_EPISODE_STEPS
        truncated = False # Truncation is not used in this logic
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_physics(self):
        # Interaction rules: RED attracts BLUE, GREEN repels YELLOW
        for i in range(len(self.clouds)):
            for j in range(i + 1, len(self.clouds)):
                c1, c2 = self.clouds[i], self.clouds[j]
                if c1 == self.selected_cloud or c2 == self.selected_cloud:
                    continue
                
                dist_vec = c2.pos - c1.pos
                dist = dist_vec.length()
                if dist == 0: continue

                force_magnitude = 0
                c1_name = self._get_color_name(c1.color)
                c2_name = self._get_color_name(c2.color)

                if (c1_name == "RED" and c2_name == "BLUE") or (c1_name == "BLUE" and c2_name == "RED"):
                    force_magnitude = -1 / dist # Attraction
                elif (c1_name == "GREEN" and c2_name == "YELLOW") or (c1_name == "YELLOW" and c2_name == "GREEN"):
                    force_magnitude = 1 / dist # Repulsion
                
                force = dist_vec.normalize() * force_magnitude * self.PHYSICS_FORCE_MULTIPLIER
                c1.apply_force(-force)
                c2.apply_force(force)

    def _handle_merges(self):
        merged_indices = set()
        reward = 0
        
        # Create a copy to iterate over while modifying the original list
        clouds_to_check = self.clouds[:]
        
        for i in range(len(clouds_to_check)):
            for j in range(i + 1, len(clouds_to_check)):
                c1 = clouds_to_check[i]
                c2 = clouds_to_check[j]

                # Ensure clouds haven't been merged already in this frame
                if c1.id in [c.id for c in self.clouds if i in merged_indices] or \
                   c2.id in [c.id for c in self.clouds if j in merged_indices]:
                    continue

                if c1.color == c2.color:
                    dist = c1.pos.distance_to(c2.pos)
                    if dist < (c1.radius + c2.radius) * self.MERGE_DISTANCE_FACTOR:
                        # Check if merge is inside a target region
                        for target in self.target_pattern:
                            if self._get_color_name(c1.color) == target["color_name"]:
                                if c1.pos.distance_to(target["pos"]) < target["radius"] or \
                                   c2.pos.distance_to(target["pos"]) < target["radius"]:
                                    reward += 5 # Merge in target region
                                    self.score += 5
                                    break
                        
                        # Find the actual cloud objects in the self.clouds list
                        main_cloud = next((c for c in self.clouds if c.id == c1.id), None)
                        other_cloud = next((c for c in self.clouds if c.id == c2.id), None)

                        if main_cloud and other_cloud:
                            main_cloud.merge(other_cloud)
                            # Mark the other cloud for removal by its id
                            merged_indices.add(other_cloud.id)
        
        if merged_indices:
            self.clouds = [c for c in self.clouds if c.id not in merged_indices]
            if self.selected_cloud and self.selected_cloud.id in merged_indices:
                self.selected_cloud = None
        return reward

    def _check_win_condition(self):
        reward = 0
        all_complete = True
        for target in self.target_pattern:
            if target["completed"]:
                continue
            
            target_color_name = target["color_name"]
            is_region_complete = False
            for cloud in self.clouds:
                if self._get_color_name(cloud.color) == target_color_name:
                    if cloud.pos.distance_to(target["pos"]) < target["radius"]:
                        # A single large cloud of the correct color fills the region
                        if len([c for c in self.clouds if self._get_color_name(c.color) == target_color_name]) == 1:
                           is_region_complete = True
                           break
            
            if is_region_complete:
                if not target["completed"]:
                    target["completed"] = True
                    reward += 10 # First time completing this region
                    self.score += 10
                    # sfx: region_complete
            else:
                all_complete = False
        
        return reward, all_complete

    def _get_cloud_target_distances(self):
        distances = {}
        for cloud in self.clouds:
            cloud_color_name = self._get_color_name(cloud.color)
            min_dist = float('inf')
            for target in self.target_pattern:
                if target["color_name"] == cloud_color_name:
                    dist = cloud.pos.distance_to(target["pos"])
                    if dist < min_dist:
                        min_dist = dist
            distances[cloud.id] = min_dist
        return distances

    def _get_color_name(self, color_tuple):
        for name, rgb in self.GAS_COLORS.items():
            if rgb == color_tuple:
                return name
        return None

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw stars
        for pos, size, color in self.stars:
            pygame.draw.circle(self.screen, color, pos, size)
        
        # Draw target patterns
        for target in self.target_pattern:
            color = self.GAS_COLORS[target["color_name"]]
            alpha = 60 if not target["completed"] else 120
            pygame.gfxdraw.filled_circle(self.screen, int(target["pos"].x), int(target["pos"].y), target["radius"], color + (alpha,))
            pygame.gfxdraw.aacircle(self.screen, int(target["pos"].x), int(target["pos"].y), target["radius"], color + (alpha + 60,))
        
        # Draw clouds
        for cloud in self.clouds:
            cloud.draw(self.screen)
        
        # Draw cursor
        if self.selected_cloud:
            pygame.draw.circle(self.screen, self.COLOR_CURSOR, self.cursor_pos, 12, 2)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, self.cursor_pos - pygame.Vector2(18,0), self.cursor_pos - pygame.Vector2(8,0), 2)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, self.cursor_pos + pygame.Vector2(8,0), self.cursor_pos + pygame.Vector2(18,0), 2)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, self.cursor_pos - pygame.Vector2(0,18), self.cursor_pos - pygame.Vector2(0,8), 2)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, self.cursor_pos + pygame.Vector2(0,8), self.cursor_pos + pygame.Vector2(0,18), 2)
        else:
            pygame.draw.circle(self.screen, self.COLOR_CURSOR, self.cursor_pos, 10, 1)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, self.cursor_pos - pygame.Vector2(5,0), self.cursor_pos + pygame.Vector2(5,0), 1)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, self.cursor_pos - pygame.Vector2(0,5), self.cursor_pos + pygame.Vector2(0,5), 1)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        level_text = self.font_ui.render(f"LEVEL: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))

        time_text = self.font_ui.render(f"TIME: {self.level_timer // 30:02d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH // 2 - time_text.get_width() // 2, 10))

        target_title = self.font_title.render("TARGET", True, self.COLOR_UI_TEXT)
        self.screen.blit(target_title, (self.WIDTH - 110, 40))
        for i, target in enumerate(self.target_pattern):
            color = self.GAS_COLORS[target["color_name"]]
            y_pos = 80 + i * 30
            status_color = (0, 255, 0) if target["completed"] else (255, 255, 255)
            pygame.draw.circle(self.screen, color, (self.WIDTH - 100, y_pos), 10)
            pygame.draw.circle(self.screen, status_color, (self.WIDTH - 100, y_pos), 10, 1)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "time_left_s": self.level_timer / 30,
            "unlocked_colors": len(self.unlocked_colors),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Nebula Crafter")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # --- Pygame Rendering ---
        # The observation is already a rendered surface, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Print Info ---
        if env.steps % 30 == 0:
            print(f"Step: {info['steps']}, Level: {info['level']}, Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")

        clock.tick(30) # Run at 30 FPS

    env.close()
    print("Game Over!")