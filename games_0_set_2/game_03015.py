
# Generated: 2025-08-28T06:44:45.744773
# Source Brief: brief_03015.md
# Brief Index: 3015

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use Left/Right arrows to steer your kart. Press Space to activate boost when available."
    )

    game_description = (
        "A fast-paced, retro-futuristic side-view racer. Navigate a procedurally generated neon track, "
        "outmaneuver opponents, and use your boost strategically to finish 3 laps in first place."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and game dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.TRACK_LENGTH = 5000
        self.NUM_LAPS = 3
        self.MAX_CRASHES = 5
        self.MAX_STEPS = 2500
        self.NUM_OPPONENTS = 3

        # EXACT spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)

        # Colors
        self.COLOR_BG = pygame.Color(10, 5, 20)
        self.COLOR_PLAYER = pygame.Color(0, 255, 150)
        self.COLOR_BOOST = pygame.Color(255, 255, 0)
        self.COLOR_CRASH = pygame.Color(255, 100, 0)
        self.COLOR_UI_TEXT = pygame.Color(220, 220, 255)
        self.COLOR_UI_BG = pygame.Color(40, 40, 80, 150)
        self.OPPONENT_COLORS = [pygame.Color(255, 0, 255), pygame.Color(0, 150, 255), pygame.Color(200, 50, 255)]
        self.TRACK_EDGE_COLOR = pygame.Color(30, 20, 50)

        # Game physics and parameters
        self.PLAYER_LATERAL_SPEED = 4.5
        self.PLAYER_FRICTION = 0.85
        self.PLAYER_FORWARD_SPEED = 12
        self.BOOST_MULTIPLIER = 2.0
        self.BOOST_DURATION = 60  # steps
        self.BOOST_COOLDOWN_TOTAL = 180  # steps
        self.CRASH_SPEED_PENALTY = 0.1 # Multiplier
        self.CRASH_DURATION = 30 # steps

        # Initialize state variables (will be set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.opponents = []
        self.particles = deque()
        self.track_segments = deque()
        self.camera_x = 0
        self.stars = []
        self.track_hue = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.track_hue = self.np_random.integers(0, 360)

        self.player = {
            "y": self.HEIGHT / 2,
            "vy": 0,
            "dist": 0,
            "lap": 1,
            "rank": self.NUM_OPPONENTS + 1,
            "crash_timer": 0,
            "is_boosting": False,
            "boost_timer": 0,
            "boost_cooldown": 0,
            "crash_count": 0,
        }

        self.opponents = []
        for i in range(self.NUM_OPPONENTS):
            self.opponents.append({
                "y": self.HEIGHT / 2 + self.np_random.uniform(-50, 50),
                "dist": self.np_random.uniform(-100, 0),
                "lap": 1,
                "speed_multiplier": self.np_random.uniform(0.85, 0.98),
                "target_y_offset": self.np_random.uniform(-40, 40),
                "aggression": self.np_random.uniform(0.01, 0.05)
            })

        self.camera_x = 0
        self._generate_initial_track()
        self._generate_stars()
        
        self.particles.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.01 # Small reward for surviving
        
        self._handle_input(movement, space_held)
        self._update_player()
        self._update_opponents()
        self._update_camera()
        self._update_track()
        self._update_particles()
        
        prev_ranks = [self.player["rank"]] + [o.get("rank", self.NUM_OPPONENTS + 1) for o in self.opponents]
        self._update_ranks()
        
        # Overtake reward
        if self.player["rank"] < prev_ranks[0]:
            reward += 1.0

        # Lap completion reward
        player_lap_before = self.player["lap"]
        self.player["lap"] = int(self.player["dist"] // self.TRACK_LENGTH) + 1
        if self.player["lap"] > player_lap_before:
            reward += 10.0
            # Increase opponent speed
            for o in self.opponents:
                o["speed_multiplier"] += 0.05
        
        for o in self.opponents:
             o["lap"] = int(o["dist"] // self.TRACK_LENGTH) + 1

        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.player["crash_count"] >= self.MAX_CRASHES:
                reward = -50.0
            elif self.player["lap"] > self.NUM_LAPS:
                rank_rewards = {1: 50.0, 2: 30.0, 3: 10.0, 4: 0.0}
                reward = rank_rewards.get(self.player["rank"], 0.0)
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _handle_input(self, movement, space_held):
        # Lateral movement
        if movement == 3:  # Left -> Move Up
            self.player["vy"] -= 1.0
        elif movement == 4:  # Right -> Move Down
            self.player["vy"] += 1.0
        
        # Boost activation
        if space_held and self.player["boost_cooldown"] == 0 and not self.player["is_boosting"]:
            self.player["is_boosting"] = True
            self.player["boost_timer"] = self.BOOST_DURATION
            self.player["boost_cooldown"] = self.BOOST_COOLDOWN_TOTAL
            # Sound: Boost Activate

    def _update_player(self):
        # Update boost state
        if self.player["is_boosting"]:
            self.player["boost_timer"] -= 1
            if self.player["boost_timer"] <= 0:
                self.player["is_boosting"] = False
                # Sound: Boost Deactivate
        elif self.player["boost_cooldown"] > 0:
            self.player["boost_cooldown"] -= 1

        # Forward speed
        boost_factor = self.BOOST_MULTIPLIER if self.player["is_boosting"] else 1.0
        crash_factor = self.CRASH_SPEED_PENALTY if self.player["crash_timer"] > 0 else 1.0
        forward_speed = self.PLAYER_FORWARD_SPEED * boost_factor * crash_factor
        self.player["dist"] += forward_speed
        
        # Lateral movement
        self.player["vy"] = max(-self.PLAYER_LATERAL_SPEED, min(self.PLAYER_LATERAL_SPEED, self.player["vy"]))
        self.player["y"] += self.player["vy"]
        self.player["vy"] *= self.PLAYER_FRICTION

        # Collision detection
        track_y, track_width = self._get_track_props_at(self.camera_x + self.WIDTH * 0.25)
        half_width = track_width / 2
        
        if not (track_y - half_width < self.player["y"] < track_y + half_width) and self.player["crash_timer"] == 0:
            self.player["crash_count"] += 1
            self.player["crash_timer"] = self.CRASH_DURATION
            self.player["vy"] *= -0.8 # Bounce off wall
            # Sound: Crash
            for _ in range(30):
                self._emit_particle(
                    pos=[self.WIDTH * 0.25, self.player["y"]],
                    vel=[self.np_random.uniform(-5, 5), self.np_random.uniform(-5, 5)],
                    lifespan=20,
                    radius_start=4,
                    radius_end=0,
                    color=self.COLOR_CRASH
                )
        
        if self.player["crash_timer"] > 0:
            self.player["crash_timer"] -= 1
        
        # Boost particles
        if self.player["is_boosting"]:
            self._emit_particle(
                pos=[self.WIDTH * 0.25 - 15, self.player["y"]],
                vel=[-8, self.np_random.uniform(-0.5, 0.5)],
                lifespan=15,
                radius_start=5,
                radius_end=0,
                color=self.COLOR_BOOST
            )
            
    def _update_opponents(self):
        for o in self.opponents:
            o["dist"] += self.PLAYER_FORWARD_SPEED * o["speed_multiplier"]
            
            track_y, _ = self._get_track_props_at(o["dist"])
            target_y = track_y + o["target_y_offset"]
            
            o["y"] += (target_y - o["y"]) * o["aggression"]
            o["y"] = max(0, min(self.HEIGHT, o["y"]))

    def _update_camera(self):
        self.camera_x = self.player["dist"] - self.WIDTH * 0.25
    
    def _update_track(self):
        # Remove old segments
        while self.track_segments and self.track_segments[0][0] < self.camera_x - 50:
            self.track_segments.popleft()
        # Add new segments
        while self.track_segments and self.track_segments[-1][0] < self.camera_x + self.WIDTH + 50:
            self._generate_track_segment()

    def _update_particles(self):
        for p in list(self.particles):
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] += 1
            if p["life"] >= p["lifespan"]:
                self.particles.remove(p)

    def _update_ranks(self):
        racers = [{"id": "player", "dist": self.player["dist"]}]
        for i, o in enumerate(self.opponents):
            racers.append({"id": f"opp_{i}", "dist": o["dist"]})
        
        racers.sort(key=lambda r: r["dist"], reverse=True)
        
        for rank, racer in enumerate(racers, 1):
            if racer["id"] == "player":
                self.player["rank"] = rank
            else:
                idx = int(racer["id"].split("_")[1])
                self.opponents[idx]["rank"] = rank

    def _check_termination(self):
        if self.player["lap"] > self.NUM_LAPS:
            self.game_over = True
        if self.player["crash_count"] >= self.MAX_CRASHES:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_track()
        self._render_opponents()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for star in self.stars:
            x = (star["x"] - self.camera_x * star["parallax"]) % self.WIDTH
            pygame.draw.circle(self.screen, star["color"], (int(x), int(star["y"])), star["size"])

    def _render_track(self):
        self.track_hue = (self.track_hue + 0.2) % 360
        track_color = pygame.Color(0,0,0)
        track_color.hsva = (self.track_hue, 80, 90, 100)
        
        points_upper = []
        points_lower = []
        points_road = []

        for x, y_center, width in self.track_segments:
            render_x = int(x - self.camera_x)
            half_width = width / 2
            points_upper.append((render_x, y_center - half_width))
            points_lower.append((render_x, y_center + half_width))
            points_road.append((render_x, y_center, width))

        if len(points_road) > 1:
            for i in range(len(points_road) - 1):
                p1_x, p1_y, p1_w = points_road[i]
                p2_x, p2_y, p2_w = points_road[i+1]
                
                # Road surface
                poly_points = [
                    (p1_x, p1_y - p1_w/2), (p2_x, p2_y - p2_w/2),
                    (p2_x, p2_y + p2_w/2), (p1_x, p1_y + p1_w/2)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, poly_points, track_color)
                pygame.gfxdraw.aapolygon(self.screen, poly_points, track_color)

                # Edges
                edge_width = 8
                edge_poly_upper = [
                    (p1_x, p1_y - p1_w/2), (p2_x, p2_y - p2_w/2),
                    (p2_x, p2_y - p2_w/2 - edge_width), (p1_x, p1_y - p1_w/2 - edge_width)
                ]
                edge_poly_lower = [
                    (p1_x, p1_y + p1_w/2), (p2_x, p2_y + p2_w/2),
                    (p2_x, p2_y + p2_w/2 + edge_width), (p1_x, p1_y + p1_w/2 + edge_width)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, edge_poly_upper, self.TRACK_EDGE_COLOR)
                pygame.gfxdraw.filled_polygon(self.screen, edge_poly_lower, self.TRACK_EDGE_COLOR)

    def _render_player(self):
        x = self.WIDTH * 0.25
        y = self.player["y"]
        
        # Glow effect
        glow_radius = 20
        if self.player["is_boosting"]:
            glow_radius = 35
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        glow_color = self.COLOR_PLAYER if not self.player["crash_timer"] > 0 else self.COLOR_CRASH
        alpha = 100 if not self.player["is_boosting"] else 150
        pygame.draw.circle(glow_surf, (*glow_color[:3], alpha), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (int(x - glow_radius), int(y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Kart body
        kart_points = [
            (x + 15, y),
            (x - 10, y - 8),
            (x - 15, y),
            (x - 10, y + 8)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, kart_points, glow_color)
        pygame.gfxdraw.aapolygon(self.screen, kart_points, glow_color)
        
    def _render_opponents(self):
        for i, o in enumerate(self.opponents):
            render_x = o["dist"] - self.camera_x
            if 0 < render_x < self.WIDTH:
                x, y = render_x, o["y"]
                color = self.OPPONENT_COLORS[i % len(self.OPPONENT_COLORS)]
                kart_points = [
                    (x + 15, y), (x - 10, y - 8),
                    (x - 15, y), (x - 10, y + 8)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, kart_points, color)
                pygame.gfxdraw.aapolygon(self.screen, kart_points, color)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p["life"] / p["lifespan"]
            current_radius = int(p["radius_start"] * (1 - life_ratio) + p["radius_end"] * life_ratio)
            if current_radius > 0:
                render_x = p["pos"][0] - self.camera_x
                pygame.draw.circle(self.screen, p["color"], (int(render_x), int(p["pos"][1])), current_radius)

    def _render_ui(self):
        # UI Background
        ui_surf = pygame.Surface((self.WIDTH, 60), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, self.HEIGHT - 60))

        # Lap Text
        lap_text = f"LAP: {min(self.player['lap'], self.NUM_LAPS)}/{self.NUM_LAPS}"
        lap_surf = self.font_small.render(lap_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(lap_surf, (15, self.HEIGHT - 45))

        # Position Text
        rank_str = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(self.player["rank"], f"{self.player['rank']}th")
        pos_text = f"POS: {rank_str}"
        pos_surf = self.font_large.render(pos_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(pos_surf, (150, self.HEIGHT - 52))

        # Boost Bar
        boost_ratio = 1.0 - (self.player["boost_cooldown"] / self.BOOST_COOLDOWN_TOTAL)
        bar_width = 200
        bar_height = 20
        bar_x, bar_y = self.WIDTH - bar_width - 80, self.HEIGHT - 45
        
        pygame.draw.rect(self.screen, self.TRACK_EDGE_COLOR, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        if boost_ratio > 0:
            fill_color = self.COLOR_BOOST if boost_ratio == 1.0 else (150,150,0)
            pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, bar_width * boost_ratio, bar_height), border_radius=5)
        boost_text = self.font_small.render("BOOST", True, self.COLOR_UI_TEXT)
        self.screen.blit(boost_text, (bar_x - 65, self.HEIGHT - 45))

        # Crash Indicator
        crash_text = f"DMG: {self.player['crash_count']}/{self.MAX_CRASHES}"
        crash_color = self.COLOR_CRASH if self.player['crash_count'] / self.MAX_CRASHES >= 0.6 else self.COLOR_UI_TEXT
        crash_surf = self.font_small.render(crash_text, True, crash_color)
        self.screen.blit(crash_surf, (self.WIDTH - 110, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": self.player["lap"],
            "rank": self.player["rank"],
            "crashes": self.player["crash_count"],
        }
    
    def _generate_initial_track(self):
        self.track_segments.clear()
        x = -50
        y = self.HEIGHT / 2
        width = 180
        dy = 0
        ddy = 0

        for _ in range(int((self.WIDTH + 100) / 10)):
            self.track_segments.append((x, y, width))
            
            ddy += self.np_random.uniform(-0.05, 0.05)
            ddy = np.clip(ddy, -0.2, 0.2)
            dy += ddy
            dy = np.clip(dy, -2, 2)
            y += dy
            y = np.clip(y, self.HEIGHT * 0.2, self.HEIGHT * 0.8)
            width += self.np_random.uniform(-2, 2)
            width = np.clip(width, 100, 250)
            x += 10

    def _generate_track_segment(self):
        last_x, last_y, last_width = self.track_segments[-1]
        x = last_x + 10
        
        dy = (last_y - self.track_segments[-2][1]) if len(self.track_segments) > 1 else 0
        y = last_y + dy + self.np_random.uniform(-3, 3)
        y = np.clip(y, self.HEIGHT * 0.2, self.HEIGHT * 0.8)

        width = last_width + self.np_random.uniform(-4, 4)
        width = np.clip(width, 100, 250)

        self.track_segments.append((x, y, width))

    def _get_track_props_at(self, world_x):
        for i in range(len(self.track_segments) - 1):
            p1 = self.track_segments[i]
            p2 = self.track_segments[i+1]
            if p1[0] <= world_x < p2[0]:
                ratio = (world_x - p1[0]) / (p2[0] - p1[0])
                y = p1[1] + (p2[1] - p1[1]) * ratio
                width = p1[2] + (p2[2] - p1[2]) * ratio
                return y, width
        return self.track_segments[-1][1], self.track_segments[-1][2]

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                "x": self.np_random.uniform(0, self.WIDTH),
                "y": self.np_random.uniform(0, self.HEIGHT),
                "parallax": self.np_random.uniform(0.1, 0.5),
                "size": self.np_random.choice([1, 1, 1, 2]),
                "color": random.choice([pygame.Color(50,50,80), pygame.Color(80,80,120), pygame.Color(120,120,150)])
            })

    def _emit_particle(self, pos, vel, lifespan, radius_start, radius_end, color):
        self.particles.append({
            "pos": list(pos), "vel": list(vel), "life": 0, "lifespan": lifespan,
            "radius_start": radius_start, "radius_end": radius_end, "color": color
        })

    def validate_implementation(self):
        print("Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Racer")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print("="*30)
    print(env.game_description)
    print("="*30)
    print(env.user_guide)
    print("="*30)

    while not done:
        # Human input mapping
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Match the intended FPS

    print(f"Game Over! Final Info: {info}")
    env.close()