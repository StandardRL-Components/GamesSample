
# Generated: 2025-08-27T21:08:50.528151
# Source Brief: brief_02690.md
# Brief Index: 2690

        
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
    An arcade snail racing game where the player competes against two AI opponents.
    The goal is to finish the race in the best possible position before time runs out.
    Players can move vertically, accelerate, brake, and use a limited speed boost.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: → to accelerate, ← to brake, ↑↓ to move vertically. "
        "Hold Space for a speed boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade snail racer. Dodge obstacles, use your boost strategically, "
        "and race to the finish line against two opponents!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG_SKY = (135, 206, 235)
    COLOR_BG_HILLS_1 = (34, 139, 34)
    COLOR_BG_HILLS_2 = (0, 100, 0)
    COLOR_TRACK = (124, 252, 0)
    COLOR_PLAYER = (255, 69, 0)
    COLOR_PLAYER_GLOW = (255, 165, 0)
    COLOR_AI_1 = (65, 105, 225)
    COLOR_AI_2 = (255, 215, 0)
    COLOR_OBSTACLE = (139, 69, 19)
    COLOR_FINISH_LINE_1 = (255, 255, 255)
    COLOR_FINISH_LINE_2 = (50, 50, 50)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 128)
    COLOR_BOOST_BAR = (0, 255, 255)
    COLOR_BOOST_EMPTY = (70, 70, 70)

    # Game Parameters
    TRACK_LENGTH = 6000
    TIME_LIMIT_SECONDS = 120
    NUM_OBSTACLES = 40
    NUM_AI_OPPONENTS = 2

    # Player Physics
    PLAYER_ACCEL = 0.05
    PLAYER_BRAKE = 0.1
    PLAYER_MAX_SPEED = 5.0
    PLAYER_VERTICAL_SPEED = 3.0
    PLAYER_DRAG = 0.9
    BOOST_AMOUNT = 5.0
    BOOST_DRAIN = 4.0
    BOOST_RECHARGE = 1.0
    MAX_BOOST = 100.0
    
    # AI Physics
    AI_INITIAL_SPEED_MIN = 3.5
    AI_INITIAL_SPEED_MAX = 4.5
    AI_DIFFICULTY_SCALING_INTERVAL = 30 # seconds
    AI_DIFFICULTY_SCALING_AMOUNT = 0.1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 18)
        self.font_large = pygame.font.SysFont("Arial", 24)
        
        self.player = {}
        self.ai_opponents = []
        self.obstacles = []
        self.particles = []
        self.finish_line = {}
        self.camera_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.results = []
        self.last_player_x = 0
        self.overtaken_opponents = set()
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_x = 0
        self.results = []
        self.overtaken_opponents = set()
        self.particles.clear()

        # Player
        self.player = {
            "id": "player",
            "x": 50.0,
            "y": self.SCREEN_HEIGHT / 2,
            "speed": 0.0,
            "target_speed": 0.0,
            "vy": 0.0,
            "boost_meter": self.MAX_BOOST,
            "collision_timer": 0,
            "color": self.COLOR_PLAYER,
            "finish_time": -1
        }
        self.last_player_x = self.player["x"]

        # AI Opponents
        self.ai_opponents = []
        ai_colors = [self.COLOR_AI_1, self.COLOR_AI_2]
        for i in range(self.NUM_AI_OPPONENTS):
            self.ai_opponents.append({
                "id": f"ai_{i}",
                "x": 40.0 - i * 10,
                "y": self.np_random.uniform(self.SCREEN_HEIGHT * 0.3, self.SCREEN_HEIGHT * 0.7),
                "speed": self.np_random.uniform(self.AI_INITIAL_SPEED_MIN, self.AI_INITIAL_SPEED_MAX),
                "base_speed": self.np_random.uniform(self.AI_INITIAL_SPEED_MIN, self.AI_INITIAL_SPEED_MAX),
                "collision_timer": 0,
                "color": ai_colors[i],
                "finish_time": -1
            })

        # Obstacles
        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            self.obstacles.append({
                "x": self.np_random.uniform(self.SCREEN_WIDTH, self.TRACK_LENGTH - self.SCREEN_WIDTH),
                "y": self.np_random.uniform(self.SCREEN_HEIGHT * 0.2, self.SCREEN_HEIGHT * 0.8),
                "radius": self.np_random.uniform(10, 20)
            })
            
        # Finish Line
        self.finish_line = {"x": self.TRACK_LENGTH}

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = self.game_over

        if not terminated:
            # --- Handle Player Input ---
            # Horizontal movement
            if movement == 4: # Right
                self.player["target_speed"] = min(self.PLAYER_MAX_SPEED, self.player["target_speed"] + self.PLAYER_ACCEL)
            elif movement == 3: # Left
                self.player["target_speed"] = max(0, self.player["target_speed"] - self.PLAYER_BRAKE)
            
            # Vertical movement
            if movement == 1: # Up
                self.player["vy"] = -self.PLAYER_VERTICAL_SPEED
            elif movement == 2: # Down
                self.player["vy"] = self.PLAYER_VERTICAL_SPEED

            # Boost
            is_boosting = False
            if space_held and self.player["boost_meter"] > 0:
                is_boosting = True
                self.player["boost_meter"] = max(0, self.player["boost_meter"] - self.BOOST_DRAIN)
                # sfx: boost_sound()
            else:
                self.player["boost_meter"] = min(self.MAX_BOOST, self.player["boost_meter"] + self.BOOST_RECHARGE)

            # --- Update Game Physics ---
            # Update player speed (lerp for smoothness)
            boost_effect = self.BOOST_AMOUNT if is_boosting else 0
            current_max_speed = self.PLAYER_MAX_SPEED + boost_effect
            self.player["speed"] = self.player["speed"] * 0.9 + min(self.player["target_speed"], current_max_speed) * 0.1
            
            if self.player["collision_timer"] > 0:
                self.player["speed"] *= 0.8 # Slow down when hit
                self.player["collision_timer"] -= 1

            # Update positions
            self.player["x"] += self.player["speed"]
            self.player["y"] += self.player["vy"]
            self.player["vy"] *= self.PLAYER_DRAG # Apply vertical drag
            self.player["y"] = np.clip(self.player["y"], self.SCREEN_HEIGHT * 0.2, self.SCREEN_HEIGHT * 0.8)

            # Create boost particles
            if is_boosting:
                self._create_particle(self.player["x"], self.player["y"], 15, self.COLOR_BOOST_BAR)

            # Update AI
            self._update_ai()

            # --- Collisions ---
            player_collided = self._check_collisions(self.player)
            for ai in self.ai_opponents:
                self._check_collisions(ai)
            
            # --- Reward Calculation ---
            # Reward for moving forward
            reward += (self.player["x"] - self.last_player_x) * 0.1
            self.last_player_x = self.player["x"]

            # Penalty for collision
            if player_collided:
                reward -= 0.02

            # Reward for overtaking
            for i, ai in enumerate(self.ai_opponents):
                if ai["id"] not in self.overtaken_opponents and self.player["x"] > ai["x"]:
                    reward += 5.0
                    self.overtaken_opponents.add(ai["id"])
                    # sfx: overtake_chime()

            # --- Check Termination ---
            all_snails = [self.player] + self.ai_opponents
            for snail in all_snails:
                if snail["x"] >= self.TRACK_LENGTH and snail["finish_time"] == -1:
                    snail["finish_time"] = self.steps
                    self.results.append(snail["id"])
            
            player_finished = self.player["finish_time"] != -1
            all_finished = all(s["finish_time"] != -1 for s in all_snails)
            time_up = self.steps >= self.TIME_LIMIT_SECONDS * self.FPS
            
            if time_up or player_finished or all_finished:
                terminated = True
                self.game_over = True
                
                # Assign terminal rewards
                if player_finished:
                    rank = self.results.index("player")
                    if rank == 0: reward += 100 # 1st
                    elif rank == 1: reward += 50 # 2nd
                    else: reward += 25 # 3rd
                elif time_up:
                    reward -= 100 # Timeout

        # Update global state
        self.steps += 1
        self.score += reward
        self._update_particles()
        self.camera_x = self.player["x"] - self.SCREEN_WIDTH / 4

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ai(self):
        # Scale difficulty over time
        if self.steps > 0 and self.steps % (self.AI_DIFFICULTY_SCALING_INTERVAL * self.FPS) == 0:
            for ai in self.ai_opponents:
                ai["base_speed"] += self.AI_DIFFICULTY_SCALING_AMOUNT

        for ai in self.ai_opponents:
            if ai["collision_timer"] > 0:
                ai["speed"] *= 0.9
                ai["collision_timer"] -= 1
            else:
                ai["speed"] = ai["base_speed"]
            
            ai["x"] += ai["speed"]
            # Add slight vertical oscillation
            ai["y"] += self.np_random.uniform(-0.5, 0.5)
            ai["y"] = np.clip(ai["y"], self.SCREEN_HEIGHT * 0.2, self.SCREEN_HEIGHT * 0.8)

    def _check_collisions(self, snail):
        collided = False
        for obs in self.obstacles:
            dist = math.hypot(snail["x"] - obs["x"], snail["y"] - obs["y"])
            if dist < obs["radius"] + 15: # 15 is snail radius
                if snail["collision_timer"] == 0:
                    snail["collision_timer"] = 15 # Stun for 0.5s
                    self._create_particle(snail["x"], snail["y"], 20, self.COLOR_OBSTACLE, is_hit_spark=True)
                    # sfx: collision_thud()
                collided = True
        return collided

    def _create_particle(self, x, y, count, color, is_hit_spark=False):
        for _ in range(count):
            if is_hit_spark:
                vel = pygame.Vector2(self.np_random.uniform(-4, 4), self.np_random.uniform(-4, 4))
                life = self.np_random.integers(10, 20)
            else: # Boost trail
                vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
                life = self.np_random.integers(20, 40)
            self.particles.append({"pos": pygame.Vector2(x, y), "vel": vel, "life": life, "color": color})
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["life"] -= 1

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_x": self.player["x"],
            "player_speed": self.player["speed"],
            "boost_meter": self.player["boost_meter"],
            "rank": self.results.index("player") + 1 if "player" in self.results else -1
        }
        
    def _render_game(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG_SKY)
        # Parallax Hills
        self._draw_hills(self.COLOR_BG_HILLS_2, 0.2, 150, 100)
        self._draw_hills(self.COLOR_BG_HILLS_1, 0.4, 200, 50)
        
        # --- Track ---
        track_rect = pygame.Rect(0, self.SCREEN_HEIGHT * 0.2, self.SCREEN_WIDTH, self.SCREEN_HEIGHT * 0.6)
        pygame.draw.rect(self.screen, self.COLOR_TRACK, track_rect)

        # --- Game Elements (relative to camera) ---
        # Finish Line
        finish_x = self.finish_line["x"] - self.camera_x
        if 0 < finish_x < self.SCREEN_WIDTH:
            for i in range(10):
                color = self.COLOR_FINISH_LINE_1 if i % 2 == 0 else self.COLOR_FINISH_LINE_2
                rect = pygame.Rect(finish_x, track_rect.top + i * track_rect.height / 10, 20, track_rect.height / 10)
                pygame.draw.rect(self.screen, color, rect)

        # Obstacles
        for obs in self.obstacles:
            ox, oy = int(obs["x"] - self.camera_x), int(obs["y"])
            if 0 < ox < self.SCREEN_WIDTH:
                pygame.gfxdraw.filled_circle(self.screen, ox, oy, int(obs["radius"]), self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.screen, ox, oy, int(obs["radius"]), (0,0,0))

        # Particles
        for p in self.particles:
            px, py = int(p["pos"].x - self.camera_x), int(p["pos"].y)
            alpha = max(0, min(255, int(255 * (p["life"] / 40.0))))
            size = max(1, int(p["life"] / 8))
            pygame.gfxdraw.filled_circle(self.screen, px, py, size, p["color"] + (alpha,))

        # Snails
        all_snails = sorted([self.player] + self.ai_opponents, key=lambda s: s["y"])
        for snail in all_snails:
            self._draw_snail(snail)
            
        # --- UI Overlay ---
        self._render_ui()

    def _draw_hills(self, color, parallax_factor, amplitude, frequency):
        points = []
        for x in range(self.SCREEN_WIDTH + 1):
            offset_x = (x + self.camera_x * parallax_factor)
            y = self.SCREEN_HEIGHT * 0.4 + math.sin(offset_x / frequency) * amplitude
            points.append((x, y))
        points.append((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        points.append((0, self.SCREEN_HEIGHT))
        pygame.draw.polygon(self.screen, color, points)
    
    def _draw_snail(self, snail):
        sx, sy = int(snail["x"] - self.camera_x), int(snail["y"])
        
        # Glow for player
        if snail["id"] == "player":
            for i in range(10, 0, -2):
                alpha = 50 - i * 5
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, 20 + i, self.COLOR_PLAYER_GLOW + (alpha,))

        # Body
        body_stretch = 1.0 + (snail.get("speed", 0) / self.PLAYER_MAX_SPEED) * 0.2
        body_rect = pygame.Rect(sx - 15, sy + 5, 30 * body_stretch, 10)
        pygame.draw.ellipse(self.screen, snail["color"], body_rect)
        
        # Shell
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, 15, (188, 143, 143))
        pygame.gfxdraw.aacircle(self.screen, sx, sy, 15, (101, 67, 33))
        
        # Eyes
        eye_x_offset = 12 * body_stretch
        pygame.draw.line(self.screen, (0,0,0), (sx, sy), (sx + eye_x_offset, sy-15), 2)
        pygame.draw.line(self.screen, (0,0,0), (sx, sy), (sx + eye_x_offset-5, sy-18), 2)
        pygame.gfxdraw.filled_circle(self.screen, int(sx + eye_x_offset), sy - 15, 4, (255,255,255))
        pygame.gfxdraw.filled_circle(self.screen, int(sx + eye_x_offset-5), sy - 18, 4, (255,255,255))
        pygame.gfxdraw.filled_circle(self.screen, int(sx + eye_x_offset), sy - 15, 2, (0,0,0))
        pygame.gfxdraw.filled_circle(self.screen, int(sx + eye_x_offset-5), sy - 18, 2, (0,0,0))

    def _render_ui(self):
        # UI Background panels
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.SCREEN_WIDTH, 40))
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, self.SCREEN_HEIGHT - 30, self.SCREEN_WIDTH, 30))

        # Timer
        time_left = max(0, (self.TIME_LIMIT_SECONDS * self.FPS - self.steps) / self.FPS)
        timer_text = self.font_large.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 8))
        
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score:.0f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 8))

        # Race Position
        all_snails = sorted([self.player] + self.ai_opponents, key=lambda s: s["x"], reverse=True)
        pos_text = "POS: "
        for i, snail in enumerate(all_snails):
            color = snail["color"]
            if snail["id"] == "player":
                pos_text += f"[{i+1}] "
            else:
                pos_text += f"{i+1} "
        pos_surf = self.font_large.render(pos_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(pos_surf, (self.SCREEN_WIDTH/2 - pos_surf.get_width()/2, 8))

        # Boost Meter
        boost_width = 150
        boost_rect_bg = pygame.Rect(10, self.SCREEN_HEIGHT - 22, boost_width, 15)
        pygame.draw.rect(self.screen, self.COLOR_BOOST_EMPTY, boost_rect_bg, border_radius=3)
        boost_fill_width = int(boost_width * (self.player["boost_meter"] / self.MAX_BOOST))
        boost_rect_fill = pygame.Rect(10, self.SCREEN_HEIGHT - 22, boost_fill_width, 15)
        pygame.draw.rect(self.screen, self.COLOR_BOOST_BAR, boost_rect_fill, border_radius=3)
        
        # Progress Bar
        progress_width = self.SCREEN_WIDTH - boost_width - 30
        progress_x = boost_width + 20
        progress_bg = pygame.Rect(progress_x, self.SCREEN_HEIGHT - 22, progress_width, 15)
        pygame.draw.rect(self.screen, self.COLOR_BOOST_EMPTY, progress_bg, border_radius=3)
        
        player_progress = self.player["x"] / self.TRACK_LENGTH
        progress_fill = pygame.Rect(progress_x, self.SCREEN_HEIGHT - 22, int(progress_width * player_progress), 15)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, progress_fill, border_radius=3)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Snail Racer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Get user input
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False
                total_reward = 0

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                print(f"--- Game Over ---")
                print(f"Final Score: {info['score']:.2f}")
                print(f"Final Rank: {info['rank']}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(GameEnv.FPS)

    env.close()