import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:53:25.615746
# Source Brief: brief_00140.md
# Brief Index: 140
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, life, angle_spread=360, min_speed=1, max_speed=3):
        angle = math.radians(random.uniform(0, angle_spread))
        speed = random.uniform(min_speed, max_speed)
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.color = color
        self.life = life
        self.initial_life = life

    def update(self):
        self.pos += self.vel
        self.vel *= 0.98  # friction
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.initial_life))
            alpha = max(0, min(255, alpha))
            radius = int(3 * (self.life / self.initial_life))
            if radius > 0:
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*self.color, alpha), (radius, radius), radius)
                surface.blit(temp_surf, (int(self.pos.x - radius), int(self.pos.y - radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a celestial agent through hazardous star systems. Match your color to collect gems and avoid rotating energy beams."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to cycle your color to match gems. Press shift to shrink and dodge lasers (once unlocked)."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    PLAYER_SPEED = 5
    PLAYER_RADIUS_NORMAL = 12
    PLAYER_RADIUS_SMALL = 7
    LASER_BASE_SPEED = 0.5
    MAX_STEPS = 2000

    # --- COLORS ---
    COLOR_BG = (10, 20, 40)
    COLOR_WHITE = (240, 240, 240)
    COLOR_RED = (255, 50, 50)
    COLOR_LASER_GLOW = (150, 20, 20)
    
    SUIT_COLORS = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Yellow
        (50, 255, 50),  # Green
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_puzzle = pygame.font.SysFont("Consolas", 30, bold=True)
        
        # Persistent state across episodes
        self.planet_level = 1
        self.unlocked_abilities = {"shrink": False}
        self.puzzle_words = ["SHRINK", "SPEED", "SHIELD", "VISION", "PORTAL", "CLOAK"]

        self.reset()
        
        # This check is for development and can be removed in production
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_color_index = 0
        self.player_is_small = False
        self.player_radius = self.PLAYER_RADIUS_NORMAL

        # Game objects
        self.gems = []
        self.lasers = []
        self.particles = []
        
        # Internal logic state
        self.prev_space_held = False
        self.prev_shift_held = False
        self.prev_dist_to_gem = float('inf')
        self.prev_dist_to_laser = float('inf')
        
        # Puzzle state
        self.gems_collected_count = 0
        self.puzzle_message = ""
        self.puzzle_message_timer = 0
        
        self._generate_starfield()
        self._generate_planet()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        self._store_pre_update_state()
        self._handle_input(action)
        self._update_game_logic()
        
        event_reward = self._check_events()
        reward += event_reward
        
        reward += self._calculate_shaping_reward()
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if not self.gems and not self.game_over:
            terminated = True
            reward += 100  # Planet complete reward
            
            # Unlock ability based on planet level
            if self.planet_level -1 < len(self.puzzle_words):
                puzzle_word = self.puzzle_words[self.planet_level-1]
                if puzzle_word == "SHRINK" and not self.unlocked_abilities["shrink"]:
                    self.unlocked_abilities["shrink"] = True
                    reward += 5 # Puzzle solve reward
                    self.puzzle_message = "ABILITY UNLOCKED: SHRINK"
                    self.puzzle_message_timer = 90 # 3 seconds at 30fps
            
            self.planet_level += 1

        if terminated and self.game_over:
             reward -= 100 # Laser hit penalty
        
        self.score += event_reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_starfield(self):
        self.stars = []
        for _ in range(150):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            size = random.uniform(0.5, 1.5)
            brightness = random.randint(50, 120)
            self.stars.append((x, y, size, (brightness, brightness, brightness)))

    def _generate_planet(self):
        num_gems = min(3 + self.planet_level, 8)
        num_lasers = min(1 + self.planet_level // 2, 5)
        
        # Generate Gems
        self.gems = []
        for _ in range(num_gems):
            while True:
                pos = pygame.Vector2(random.randint(50, self.WIDTH - 50), random.randint(50, self.HEIGHT - 50))
                if pos.distance_to(self.player_pos) > 100: # Don't spawn on player
                    break
            
            num_colors = min(2 + self.planet_level // 2, len(self.SUIT_COLORS))
            color_idx = random.randint(0, num_colors - 1)
            self.gems.append({"pos": pos, "color_index": color_idx, "angle": random.uniform(0, 360)})

        # Generate Lasers
        self.lasers = []
        for _ in range(num_lasers):
            pos = pygame.Vector2(random.randint(100, self.WIDTH - 100), random.randint(80, self.HEIGHT - 80))
            length = random.randint(80, 150)
            angle = random.uniform(0, 360)
            speed_multiplier = 1.0 + (self.planet_level * 0.1)
            speed = (random.choice([-1, 1]) * self.LASER_BASE_SPEED) * speed_multiplier
            self.lasers.append({"pos": pos, "length": length, "angle": angle, "speed": speed, "base_speed": speed})

    def _store_pre_update_state(self):
        # For shaping rewards
        # Closest gem of matching color
        matching_gems = [g for g in self.gems if g["color_index"] == self.player_color_index]
        if matching_gems:
            closest_gem = min(matching_gems, key=lambda g: self.player_pos.distance_to(g["pos"]))
            self.prev_dist_to_gem = self.player_pos.distance_to(closest_gem["pos"])
        else:
            self.prev_dist_to_gem = float('inf')

        # Closest laser beam
        if self.lasers:
            min_dist = float('inf')
            for laser in self.lasers:
                p1 = laser["pos"]
                p2 = laser["pos"] + pygame.Vector2(laser["length"], 0).rotate(laser["angle"])
                dist = self._point_line_segment_distance(self.player_pos, p1, p2)
                if dist < min_dist:
                    min_dist = dist
            self.prev_dist_to_laser = min_dist
        else:
            self.prev_dist_to_laser = float('inf')

    def _handle_input(self, action):
        movement, space_val, shift_val = action
        space_held = space_val == 1
        shift_held = shift_val == 1

        # Movement
        if movement == 1: self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3: self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos.x += self.PLAYER_SPEED

        # Boundary checks
        self.player_pos.x = max(self.player_radius, min(self.WIDTH - self.player_radius, self.player_pos.x))
        self.player_pos.y = max(self.player_radius, min(self.HEIGHT - self.player_radius, self.player_pos.y))

        # Color change (on button press, not hold)
        if space_held and not self.prev_space_held:
            self.player_color_index = (self.player_color_index + 1) % len(self.SUIT_COLORS)
            # sfx: color_change.wav
            for _ in range(10):
                self.particles.append(Particle(self.player_pos.x, self.player_pos.y, self.SUIT_COLORS[self.player_color_index], 20, 360, 0.5, 2))

        # Size change (on button press, if unlocked)
        if shift_held and not self.prev_shift_held and self.unlocked_abilities["shrink"]:
            self.player_is_small = not self.player_is_small
            self.player_radius = self.PLAYER_RADIUS_SMALL if self.player_is_small else self.PLAYER_RADIUS_NORMAL
            # sfx: shrink_toggle.wav

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_game_logic(self):
        # Update laser speed based on steps
        if self.steps > 0 and self.steps % 500 == 0:
            for laser in self.lasers:
                laser["speed"] *= 1.05

        # Update lasers
        for laser in self.lasers:
            laser["angle"] += laser["speed"]

        # Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

        # Update puzzle message timer
        if self.puzzle_message_timer > 0:
            self.puzzle_message_timer -= 1
            if self.puzzle_message_timer == 0:
                self.puzzle_message = ""

    def _check_events(self):
        reward = 0
        # Gem collection
        for gem in self.gems[:]:
            if self.player_pos.distance_to(gem["pos"]) < self.player_radius + 8:
                if gem["color_index"] == self.player_color_index:
                    reward += 10
                    self.gems.remove(gem)
                    self.gems_collected_count += 1
                    # sfx: gem_collect.wav
                    for _ in range(30):
                        self.particles.append(Particle(gem["pos"].x, gem["pos"].y, self.SUIT_COLORS[gem["color_index"]], 40))
                else:
                    # Penalty for touching wrong color gem
                    reward -= 0.1 

        # Laser collision
        for laser in self.lasers:
            p1 = laser["pos"]
            p2 = p1 + pygame.Vector2(laser["length"], 0).rotate(laser["angle"])
            dist = self._point_line_segment_distance(self.player_pos, p1, p2)
            if dist < self.player_radius:
                self.game_over = True
                # sfx: player_hit.wav
                # Create explosion effect
                for _ in range(100):
                    self.particles.append(Particle(self.player_pos.x, self.player_pos.y, self.COLOR_RED, 60, 360, 1, 5))
                break
        
        return reward
    
    def _point_line_segment_distance(self, p, a, b):
        # p, a, and b are pygame.Vector2
        if a == b:
            return p.distance_to(a)
        
        ab = b - a
        ap = p - a
        
        ab_mag_sq = ab.magnitude_squared()
        dot_product = ap.dot(ab)
        t = dot_product / ab_mag_sq
        
        if t < 0:
            closest_point = a
        elif t > 1:
            closest_point = b
        else:
            closest_point = a + t * ab
            
        return p.distance_to(closest_point)

    def _calculate_shaping_reward(self):
        shaping_reward = 0
        
        # Reward for moving towards correct gem
        matching_gems = [g for g in self.gems if g["color_index"] == self.player_color_index]
        if matching_gems:
            closest_gem = min(matching_gems, key=lambda g: self.player_pos.distance_to(g["pos"]))
            dist_now = self.player_pos.distance_to(closest_gem["pos"])
            shaping_reward += (self.prev_dist_to_gem - dist_now) * 0.01  # Scaled reward
        
        # Penalty for moving towards any laser
        if self.lasers:
            min_dist_now = float('inf')
            for laser in self.lasers:
                p1 = laser["pos"]
                p2 = laser["pos"] + pygame.Vector2(laser["length"], 0).rotate(laser["angle"])
                dist = self._point_line_segment_distance(self.player_pos, p1, p2)
                if dist < min_dist_now:
                    min_dist_now = dist
            
            # Penalize getting closer, but not too harshly
            if min_dist_now < 100: # Only apply penalty when close
                 shaping_reward -= (self.prev_dist_to_laser - min_dist_now) * 0.02
        
        return np.clip(shaping_reward, -1, 1)

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "planet_level": self.planet_level,
            "gems_remaining": len(self.gems),
            "unlocked_shrink": self.unlocked_abilities["shrink"]
        }

    def _render_all(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        for x, y, size, color in self.stars:
            pygame.draw.circle(self.screen, color, (x, y), size)

        # Game objects
        for p in self.particles:
            p.draw(self.screen)
        
        if not self.game_over:
            for gem in self.gems:
                self._draw_gem(gem)
            self._draw_player()
        
        for laser in self.lasers:
            self._draw_laser(laser)

        # UI
        self._render_ui()
        
        # Game over flash
        if self.game_over and self.steps - self.MAX_STEPS < 2: # Show for 2 frames
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((255, 0, 0, 100))
            self.screen.blit(s, (0, 0))

    def _draw_glow(self, surface, color, center, radius, alpha_decay=0.8):
        max_radius = int(radius * 2.5)
        for i in range(max_radius, int(radius), -2):
            alpha = int(255 * (1 - (i - radius) / (max_radius - radius))**2 * alpha_decay)
            if alpha > 0:
                pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), i, (*color, alpha))

    def _draw_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        color = self.SUIT_COLORS[self.player_color_index]
        
        # Pulsating glow
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        glow_radius = self.player_radius * (1.5 + pulse * 0.5)
        self._draw_glow(self.screen, color, pos, glow_radius, 0.5)
        
        # Core
        pygame.draw.circle(self.screen, color, pos, self.player_radius)
        pygame.draw.circle(self.screen, self.COLOR_WHITE, pos, int(self.player_radius * 0.6))

    def _draw_gem(self, gem):
        pos = (int(gem["pos"].x), int(gem["pos"].y))
        color = self.SUIT_COLORS[gem["color_index"]]
        radius = 8
        
        gem["angle"] = (gem["angle"] + 2) % 360
        
        # Glow
        self._draw_glow(self.screen, color, pos, radius, 0.7)
        
        # Core crystal shape
        points = []
        for i in range(6):
            angle = math.radians(gem["angle"] + i * 60)
            r = radius if i % 2 == 0 else radius * 0.6
            points.append((pos[0] + r * math.cos(angle), pos[1] + r * math.sin(angle)))
        
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_laser(self, laser):
        p1 = laser["pos"]
        p2 = p1 + pygame.Vector2(laser["length"], 0).rotate(laser["angle"])

        # Glow
        pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, p1, p2, 7)
        # Core
        pygame.draw.line(self.screen, self.COLOR_RED, p1, p2, 3)
        pygame.draw.line(self.screen, self.COLOR_WHITE, p1, p2, 1)

        # Emitter
        pygame.draw.circle(self.screen, self.COLOR_RED, (int(p1.x), int(p1.y)), 6)
        pygame.draw.circle(self.screen, self.COLOR_WHITE, (int(p1.x), int(p1.y)), 2)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))

        # Gems collected
        gem_text = self.font_main.render(f"GEMS: {self.gems_collected_count} / {self.gems_collected_count + len(self.gems)}", True, self.COLOR_WHITE)
        self.screen.blit(gem_text, (self.WIDTH - gem_text.get_width() - 10, 10))
        
        # Planet level
        planet_text = self.font_main.render(f"PLANET: {self.planet_level}", True, self.COLOR_WHITE)
        self.screen.blit(planet_text, (self.WIDTH // 2 - planet_text.get_width() // 2, 10))

        # Current suit color indicator
        color_indicator_pos = (30, self.HEIGHT - 30)
        self._draw_glow(self.screen, self.SUIT_COLORS[self.player_color_index], color_indicator_pos, 15, 1.0)
        pygame.draw.circle(self.screen, self.SUIT_COLORS[self.player_color_index], color_indicator_pos, 15)
        
        # Shrink ability indicator
        if self.unlocked_abilities["shrink"]:
            shrink_indicator_pos = (self.WIDTH - 30, self.HEIGHT - 30)
            color = self.COLOR_WHITE if self.player_is_small else (100, 100, 120)
            pygame.draw.circle(self.screen, color, shrink_indicator_pos, 15, 2)
            pygame.draw.circle(self.screen, color, shrink_indicator_pos, 7, 2)

        # Puzzle message
        if self.puzzle_message_timer > 0:
            puzzle_surf = self.font_puzzle.render(self.puzzle_message, True, self.COLOR_WHITE)
            pos = (self.WIDTH // 2 - puzzle_surf.get_width() // 2, self.HEIGHT // 2 - puzzle_surf.get_height() // 2)
            self.screen.blit(puzzle_surf, pos)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
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

if __name__ == '__main__':
    # Example of how to run the environment
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # --- Manual Control ---
    # Use arrow keys for movement, SPACE to change color, LSHIFT to shrink
    while running:
        action = [0, 0, 0] # no-op, released, released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render for human viewing
        # Pygame uses (width, height), but our obs is (height, width, 3)
        # So we need to transpose it back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # If you have a display, you can show the window
        try:
            display_screen = pygame.display.get_surface()
            if display_screen is None:
                display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
        except pygame.error:
             print("No display available. Skipping rendering.")
             running = False # Exit if no display is found

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward}. Info: {info}")
            total_reward = 0
            obs, info = env.reset()
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()