import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Hold SPACE to charge jump power. Use ←→ to aim. Release SPACE to jump."
    )

    game_description = (
        "Hop across procedurally generated platforms to reach the goal before time runs out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Pygame Setup
        self.screen_width = 640
        self.screen_height = 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Colors
        self.COLOR_BG_TOP = (20, 20, 40)
        self.COLOR_BG_BOTTOM = (40, 30, 60)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_BOOST = (0, 255, 255)
        self.COLOR_PLATFORM_GREEN = (50, 205, 50)
        self.COLOR_PLATFORM_RED = (255, 69, 0)
        self.COLOR_PLATFORM_BLUE = (30, 144, 255)
        self.COLOR_GOAL = (255, 215, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE_DUST = (180, 180, 160)
        self.COLOR_PARTICLE_BOOST = (0, 255, 255)

        # Game Parameters
        self.FPS = 30
        self.GRAVITY = 0.4
        self.MAX_STEPS = 60 * self.FPS # 60 seconds per stage
        self.JUMP_CHARGE_RATE = 0.2
        self.MAX_JUMP_CHARGE = 10
        self.JUMP_VEL_Y = -8
        self.JUMP_VEL_X = 5
        self.BOOST_MULTIPLIER = 1.5

        # State Variables (initialized in reset)
        self.player = None
        self.platforms = None
        self.particles = None
        self.camera_x = None
        self.steps = None
        self.score = None
        self.stage = None
        self.time_left = None
        self.game_over = None
        self.jump_charge = None
        self.was_space_held = None
        self.np_random = None
        
        self.total_score = 0 # Persistent score across stages in an episode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.player = {
            "x": 100, "y": 200, "w": 20, "h": 30,
            "vx": 0, "vy": 0,
            "on_ground": False, "has_jump_boost": False
        }
        self.platforms = []
        self.particles = []
        self.camera_x = 0
        self.steps = 0
        self.score = 0
        self.total_score = 0
        self.stage = 1
        self.time_left = self.MAX_STEPS
        self.game_over = False
        self.jump_charge = 0
        self.was_space_held = False
        
        self._generate_stage()

        return self._get_observation(), self._get_info()

    def _generate_stage(self):
        self.platforms.clear()
        
        # Stage-based difficulty
        stage_params = {
            1: {"gap_x_min": 60, "gap_x_max": 120, "gap_y_max": 40, "red_prob": 0.1, "blue_prob": 0.15},
            2: {"gap_x_min": 70, "gap_x_max": 140, "gap_y_max": 50, "red_prob": 0.3, "blue_prob": 0.1},
            3: {"gap_x_min": 80, "gap_x_max": 160, "gap_y_max": 60, "red_prob": 0.6, "blue_prob": 0.05}
        }[self.stage]

        # Initial platform
        last_x, last_y = 50, 300
        self.platforms.append({"x": last_x, "y": last_y, "w": 150, "h": 20, "type": "green", "id": 0})
        self.player["x"] = last_x + 75 - self.player["w"] / 2
        self.player["y"] = last_y - self.player["h"]

        num_platforms = self.np_random.integers(10, 15)
        for i in range(1, num_platforms):
            gap_x = self.np_random.uniform(stage_params["gap_x_min"], stage_params["gap_x_max"])
            gap_y = self.np_random.uniform(-stage_params["gap_y_max"], stage_params["gap_y_max"])
            width = self.np_random.uniform(60, 120)
            
            x = self.platforms[-1]["x"] + self.platforms[-1]["w"] + gap_x
            y = np.clip(self.platforms[-1]["y"] + gap_y, 150, self.screen_height - 50)
            
            rand_val = self.np_random.random()
            if rand_val < stage_params["red_prob"]:
                ptype = "red"
            elif rand_val < stage_params["red_prob"] + stage_params["blue_prob"]:
                ptype = "blue"
            else:
                ptype = "green"

            self.platforms.append({"x": x, "y": y, "w": width, "h": 20, "type": ptype, "id": i})
            last_x = x

        # Goal platform
        goal_x = self.platforms[-1]["x"] + self.platforms[-1]["w"] + self.np_random.uniform(80, 120)
        goal_y = self.platforms[-1]["y"] + self.np_random.uniform(-40, 40)
        self.platforms.append({"x": goal_x, "y": np.clip(goal_y, 150, self.screen_height - 50), "w": 100, "h": 40, "type": "goal", "id": -1})

    def step(self, action):
        movement, space_held, _ = action
        reward = 0.01  # Small reward for surviving
        terminated = False
        
        # --- 1. Input and Jump Logic ---
        if self.player["on_ground"]:
            if space_held:
                self.jump_charge = min(self.MAX_JUMP_CHARGE, self.jump_charge + self.JUMP_CHARGE_RATE)
            elif self.was_space_held and self.jump_charge > 1: # Jump on release
                # SFX: Jump
                jump_power = self.jump_charge
                if self.player["has_jump_boost"]:
                    jump_power *= self.BOOST_MULTIPLIER
                    self.player["has_jump_boost"] = False
                    self._create_particles(self.player["x"] + self.player["w"]/2, self.player["y"] + self.player["h"], 20, self.COLOR_PARTICLE_BOOST)

                self.player["vy"] = self.JUMP_VEL_Y * (jump_power / self.MAX_JUMP_CHARGE)
                
                if movement == 3: # Left
                    self.player["vx"] = -self.JUMP_VEL_X * (jump_power / self.MAX_JUMP_CHARGE)
                elif movement == 4: # Right
                    self.player["vx"] = self.JUMP_VEL_X * (jump_power / self.MAX_JUMP_CHARGE)
                else: # Up or None - default to a slight forward movement
                    self.player["vx"] = self.JUMP_VEL_X * 0.5 * (jump_power / self.MAX_JUMP_CHARGE)

                self.player["on_ground"] = False
                self.jump_charge = 0

        if not space_held:
            self.jump_charge = 0
        self.was_space_held = bool(space_held)

        # --- 2. Physics Update ---
        if not self.player["on_ground"]:
            self.player["vy"] += self.GRAVITY
        self.player["x"] += self.player["vx"]
        self.player["y"] += self.player["vy"]

        # Dampen horizontal velocity in air
        if not self.player["on_ground"]:
            self.player["vx"] *= 0.99
        else:
            self.player["vx"] = 0

        # --- 3. Collision Detection ---
        self.player["on_ground"] = False
        px, py, pw, ph = self.player["x"], self.player["y"], self.player["w"], self.player["h"]

        for i, plat in reversed(list(enumerate(self.platforms))):
            plat_x, plat_y, plat_w, plat_h = plat["x"], plat["y"], plat["w"], plat["h"]
            
            # Manual AABB collision check using float values for precision.
            # Inclusive on the bottom edge to detect landing on a surface.
            collides_x = (px + pw > plat_x) and (px < plat_x + plat_w)
            collides_y = (py + ph >= plat_y) and (py < plat_y + plat_h)

            if collides_x and collides_y and self.player["vy"] >= 0:
                # Check if the player's feet were at or above the platform in the previous frame.
                # This prevents tunneling through the platform from below.
                previous_foot_y = py + ph - self.player["vy"]
                if previous_foot_y <= plat_y:
                    self.player["y"] = plat_y - ph
                    self.player["vy"] = 0
                    self.player["on_ground"] = True
                    
                    # SFX: Land
                    player_mid_bottom_x = self.player["x"] + self.player["w"] / 2
                    player_mid_bottom_y = self.player["y"] + self.player["h"]
                    self._create_particles(player_mid_bottom_x, player_mid_bottom_y, 10, self.COLOR_PARTICLE_DUST)
                    
                    reward += 1.0 # Landed on a platform
                    
                    if plat["type"] == "red":
                        # SFX: Crack
                        self.platforms.pop(i)
                    elif plat["type"] == "blue":
                        # SFX: Powerup
                        self.player["has_jump_boost"] = True
                        reward += 2.0
                    elif plat["type"] == "goal":
                        self.score += 100 * (self.time_left / self.MAX_STEPS)
                        self.total_score += self.score
                        if self.stage < 3:
                            # SFX: Stage Clear
                            self.stage += 1
                            self.time_left = self.MAX_STEPS
                            self._generate_stage()
                            self.camera_x = 0
                        else:
                            # SFX: Victory
                            terminated = True
                        break # For the 'for plat' loop
                    
                    break # For the 'for plat' loop, stop checking after one collision

        # --- 4. Game State Update ---
        self.time_left -= 1
        self.steps += 1
        
        target_camera_x = self.player["x"] - self.screen_width / 4
        self.camera_x += (target_camera_x - self.camera_x) * 0.1 # Smooth camera

        # Update particles
        for p in self.particles[:]:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

        # --- 5. Termination Check ---
        if self.player["y"] > self.screen_height + 50:
            # SFX: Fall
            reward = -5.0
            terminated = True
        
        if self.time_left <= 0 and not terminated: # Don't override victory
            # SFX: Time out
            terminated = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self._render_background()
        self._render_particles()
        self._render_platforms()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.total_score + self.score, "steps": self.steps, "stage": self.stage}

    def _render_background(self):
        for y in range(self.screen_height):
            mix_ratio = y / self.screen_height
            color = (
                self.COLOR_BG_TOP[0] * (1 - mix_ratio) + self.COLOR_BG_BOTTOM[0] * mix_ratio,
                self.COLOR_BG_TOP[1] * (1 - mix_ratio) + self.COLOR_BG_BOTTOM[1] * mix_ratio,
                self.COLOR_BG_TOP[2] * (1 - mix_ratio) + self.COLOR_BG_BOTTOM[2] * mix_ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))

    def _render_platforms(self):
        for plat in self.platforms:
            color_map = {
                "green": self.COLOR_PLATFORM_GREEN,
                "red": self.COLOR_PLATFORM_RED,
                "blue": self.COLOR_PLATFORM_BLUE,
                "goal": self.COLOR_GOAL
            }
            color = color_map[plat["type"]]
            
            rect = pygame.Rect(
                int(plat["x"] - self.camera_x), int(plat["y"]),
                int(plat["w"]), int(plat["h"])
            )
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in color), rect, width=2, border_radius=3)

            if plat["type"] == "goal":
                flag_pole = (rect.centerx, rect.top)
                pygame.draw.line(self.screen, self.COLOR_TEXT, (flag_pole[0], flag_pole[1]-20), flag_pole, 2)
                pygame.draw.polygon(self.screen, self.COLOR_TEXT, [(flag_pole[0], flag_pole[1]-20), (flag_pole[0]+15, flag_pole[1]-15), (flag_pole[0], flag_pole[1]-10)])

    def _render_player(self):
        squash = 0
        if self.player["on_ground"] and self.jump_charge > 0:
            squash = (self.jump_charge / self.MAX_JUMP_CHARGE) * (self.player["h"] * 0.4)
        
        player_w = self.player["w"] + squash * 0.5
        player_h = self.player["h"] - squash
        player_x = self.player["x"] - self.camera_x - (squash * 0.25)
        player_y = self.player["y"] + squash
        
        color = self.COLOR_PLAYER_BOOST if self.player["has_jump_boost"] else self.COLOR_PLAYER
        player_rect = pygame.Rect(int(player_x), int(player_y), int(player_w), int(player_h))
        
        pygame.draw.rect(self.screen, color, player_rect, border_radius=4)
        
        # Eyes
        eye_y = player_y + player_h * 0.3
        eye_x_offset = player_w * 0.2
        eye_dir = 1 if self.player["vx"] >= 0 else -1
        pygame.draw.circle(self.screen, (0,0,0), (int(player_x + player_w/2 + eye_x_offset*eye_dir), int(eye_y)), 2)

        # Jump charge meter
        if self.jump_charge > 0:
            bar_w = self.player["w"]
            bar_h = 5
            bar_x = player_x + (player_w - bar_w) / 2
            bar_y = player_y - bar_h - 5
            
            fill_w = (self.jump_charge / self.MAX_JUMP_CHARGE) * bar_w
            
            pygame.draw.rect(self.screen, (100, 100, 100), (int(bar_x), int(bar_y), int(bar_w), int(bar_h)), border_radius=2)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (int(bar_x), int(bar_y), int(fill_w), int(bar_h)), border_radius=2)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (int(p["x"] - self.camera_x - p["size"]), int(p["y"] - p["size"])))

    def _render_ui(self):
        # Stage
        stage_text = self.font_small.render(f"Stage: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Score
        score_text = self.font_small.render(f"Score: {int(self.total_score + self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(centerx=self.screen_width/2, top=10)
        self.screen.blit(score_text, score_rect)

        # Timer
        time_str = f"{self.time_left / self.FPS:.1f}"
        time_text = self.font_large.render(time_str, True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(right=self.screen_width - 10, top=5)
        self.screen.blit(time_text, time_rect)

    def _create_particles(self, x, y, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            life = self.np_random.integers(15, 31) # high is exclusive
            self.particles.append({
                "x": x, "y": y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": life, "max_life": life,
                "size": self.np_random.integers(2, 6), # high is exclusive
                "color": color
            })
            
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To run, ensure you have 'pygame' installed.
    # pip install pygame
    
    # Un-comment the line below to run in a window instead of headless
    os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Platform Hopper")
    
    terminated = False
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False
        
        if terminated:
            # Display game over message
            font = pygame.font.Font(None, 50)
            text = font.render("GAME OVER", True, (255, 0, 0))
            text_rect = text.get_rect(center=(env.screen_width/2, env.screen_height/2 - 20))
            screen.blit(text, text_rect)
            
            font_small = pygame.font.Font(None, 30)
            score_text = font_small.render(f"Final Score: {int(info['score'])}", True, (255, 255, 255))
            score_rect = score_text.get_rect(center=(env.screen_width/2, env.screen_height/2 + 20))
            screen.blit(score_text, score_rect)

            reset_text = font_small.render("Press 'R' to restart", True, (255, 255, 255))
            reset_rect = reset_text.get_rect(center=(env.screen_width/2, env.screen_height/2 + 50))
            screen.blit(reset_text, reset_rect)
            
            pygame.display.flip()
            continue

        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame
        # Convert it back to a Pygame surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # --- Frame Rate ---
        env.clock.tick(env.FPS)

    env.close()