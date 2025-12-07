
# Generated: 2025-08-28T02:25:40.084191
# Source Brief: brief_01699.md
# Brief Index: 1699

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move. ↑ to jump. Press Space for a mid-air boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced pixel-art platformer. Navigate a procedural world, jump over pits, and reach the goal before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (20, 20, 40)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (150, 200, 255)
    COLOR_PLATFORM = (100, 200, 120)
    COLOR_OBSTACLE = (255, 80, 80)
    COLOR_BOOST_PICKUP = (255, 255, 100)
    COLOR_FLAG = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)

    # Physics
    FPS = 30
    GRAVITY = 0.8
    PLAYER_JUMP_STRENGTH = -14
    PLAYER_BOOST_STRENGTH = -12
    PLAYER_MOVE_SPEED = 6
    MAX_FALL_SPEED = 15

    # Game Rules
    TIME_LIMIT_SECONDS = 60
    MAX_EPISODE_STEPS = 1000
    INITIAL_BOOSTS = 1
    STUCK_LIMIT = 150 # Frames to be considered stuck

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)
        
        # State variables (initialized in reset)
        self.player = None
        self.platforms = []
        self.obstacles = []
        self.boost_pickups = []
        self.end_flag = None
        self.particles = []
        self.camera_x = 0
        
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        self.win = False
        
        self.prev_space_held = False
        self.last_player_x = 0
        self.stuck_counter = 0
        self.difficulty_modifier = 1.0

        self.np_random = None

        # Call validation at the end of __init__
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.timer = self.TIME_LIMIT_SECONDS * self.FPS
        self.difficulty_modifier = 1.0
        
        self.particles = []
        self._generate_level()
        
        player_start_x = self.platforms[0].centerx
        player_start_y = self.platforms[0].top - 30
        self.player = {
            "rect": pygame.Rect(player_start_x, player_start_y, 24, 24),
            "vx": 0,
            "vy": 0,
            "on_ground": False,
            "boosts": self.INITIAL_BOOSTS
        }
        
        self.last_player_x = self.player["rect"].x
        self.stuck_counter = 0
        self.prev_space_held = True # Prevent boost on first frame

        self.camera_x = self.player["rect"].centerx - self.SCREEN_WIDTH / 4
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = -0.01  # Penalty for each step taken
        terminated = self.game_over

        if not terminated:
            prev_player_x_dist = abs(self.end_flag.x - self.player["rect"].x)

            self._handle_input(action)
            self._update_physics()
            reward += self._handle_collisions()
            self._update_world()
            
            # Reward for getting closer to the goal
            current_player_x_dist = abs(self.end_flag.x - self.player["rect"].x)
            reward += (prev_player_x_dist - current_player_x_dist) * 0.05

            self._check_termination()
            terminated = self.game_over
            
            if self.win:
                reward += 100
            elif self.game_over:
                reward -= 100

        self.steps += 1
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Horizontal movement
        if movement == 3: # Left
            self.player["vx"] = -self.PLAYER_MOVE_SPEED
        elif movement == 4: # Right
            self.player["vx"] = self.PLAYER_MOVE_SPEED
        else:
            self.player["vx"] = 0
        
        # Jump
        if movement == 1 and self.player["on_ground"]:
            self.player["vy"] = self.PLAYER_JUMP_STRENGTH
            self.player["on_ground"] = False
            self._spawn_particles(self.player["rect"].midbottom, 15, self.COLOR_PLATFORM)
            # sfx: jump

        # Boost
        if space_held and not self.prev_space_held and self.player["boosts"] > 0:
            self.player["vy"] = self.PLAYER_BOOST_STRENGTH
            self.player["boosts"] -= 1
            self._spawn_particles(self.player["rect"].center, 25, self.COLOR_BOOST_PICKUP, angle_range=(0, 360))
            # sfx: boost
        
        self.prev_space_held = space_held

    def _update_physics(self):
        # Update player
        self.player["vy"] += self.GRAVITY
        self.player["vy"] = min(self.player["vy"], self.MAX_FALL_SPEED)
        
        self.player["rect"].x += int(self.player["vx"])
        self.player["rect"].y += int(self.player["vy"])

        self.player["on_ground"] = False

        # Add trail particles
        if abs(self.player["vx"]) > 0 and self.player["on_ground"]:
            if self.steps % 3 == 0:
                self._spawn_particles(self.player["rect"].midbottom, 1, self.COLOR_PLATFORM, life=10, speed=1)

    def _handle_collisions(self):
        reward = 0
        player_rect = self.player["rect"]

        # Platform collisions
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                # Player falling onto platform
                if self.player["vy"] > 0 and player_rect.bottom <= plat.top + self.player["vy"] + 1:
                    player_rect.bottom = plat.top
                    self.player["vy"] = 0
                    if not self.player["on_ground"]:
                        self._spawn_particles(player_rect.midbottom, 5, self.COLOR_PLATFORM)
                        # sfx: land
                    self.player["on_ground"] = True
                # Player hitting platform from below
                elif self.player["vy"] < 0 and player_rect.top >= plat.bottom + self.player["vy"] - 1:
                    player_rect.top = plat.bottom
                    self.player["vy"] = 0
                # Player hitting side of platform
                else:
                    if self.player["vx"] > 0:
                        player_rect.right = plat.left
                    elif self.player["vx"] < 0:
                        player_rect.left = plat.right

        # Obstacle collisions
        for obs in self.obstacles:
            if player_rect.colliderect(obs["rect"]):
                self.game_over = True
                self._spawn_particles(player_rect.center, 50, self.COLOR_OBSTACLE, angle_range=(0, 360))
                # sfx: death_obstacle
                return -1 # Event-based penalty from brief

        # Boost pickup collisions
        for boost in self.boost_pickups[:]:
            if player_rect.colliderect(boost):
                self.player["boosts"] += 1
                self.boost_pickups.remove(boost)
                reward += 5
                self._spawn_particles(boost.center, 30, self.COLOR_BOOST_PICKUP, angle_range=(0, 360))
                # sfx: pickup_boost

        # End flag collision
        if player_rect.colliderect(self.end_flag):
            self.game_over = True
            self.win = True
            self._spawn_particles(player_rect.center, 100, self.COLOR_FLAG, angle_range=(0, 360), speed=8)
            # sfx: win

        return reward

    def _update_world(self):
        # Update obstacles
        for obs in self.obstacles:
            obs["rect"].x += obs["vx"]
            obs["rect"].y += obs["vy"]
            if obs["rect"].left < obs["min_x"] or obs["rect"].right > obs["max_x"]:
                obs["vx"] *= -1
            if obs["rect"].top < obs["min_y"] or obs["rect"].bottom > obs["max_y"]:
                obs["vy"] *= -1

        # Update particles
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

        # Update camera
        target_camera_x = self.player["rect"].centerx - self.SCREEN_WIDTH / 2
        # Smooth camera movement (lerp)
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

        # Update timer
        self.timer -= 1
        
        # Update difficulty
        if self.steps > 0 and self.steps % 50 == 0:
            self.difficulty_modifier *= 1.01

    def _check_termination(self):
        # Fall into pit
        if self.player["rect"].top > self.SCREEN_HEIGHT + 50:
            self.game_over = True
            # sfx: death_fall

        # Timer runs out
        if self.timer <= 0:
            self.game_over = True

        # Stuck detection
        if abs(self.player["rect"].x - self.last_player_x) < 2:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_player_x = self.player["rect"].x
        
        if self.stuck_counter >= self.STUCK_LIMIT:
            self.game_over = True

        # Max steps
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_world()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": max(0, self.timer // self.FPS),
            "boosts": self.player["boosts"] if self.player else 0,
            "win": self.win,
        }

    def _render_background(self):
        # Parallax stars
        for i in range(50):
            # Use a seeded but deterministic position for stars
            x = (i * 37 + self.steps * 0.1) % self.SCREEN_WIDTH
            y = (i * 101) % self.SCREEN_HEIGHT
            size = (i % 3) + 1
            color_val = 50 + (i % 3) * 20
            pygame.draw.rect(self.screen, (color_val, color_val, color_val + 20), (int(x), int(y), size, size))

    def _render_world(self):
        cam_x, cam_y = int(self.camera_x), 0

        # Render platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat.move(-cam_x, -cam_y))

        # Render obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs["rect"].move(-cam_x, -cam_y))

        # Render boost pickups
        for boost in self.boost_pickups:
            pygame.draw.rect(self.screen, self.COLOR_BOOST_PICKUP, boost.move(-cam_x, -cam_y))
            
        # Render end flag
        flag_rect = self.end_flag.move(-cam_x, -cam_y)
        pygame.draw.rect(self.screen, self.COLOR_FLAG, flag_rect)
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, [(flag_rect.left, flag_rect.top), (flag_rect.left, flag_rect.top + 20), (flag_rect.left - 30, flag_rect.top + 10)])

        # Render player if not terminated by falling
        if not (self.game_over and self.player["rect"].top > self.SCREEN_HEIGHT):
            player_screen_rect = self.player["rect"].move(-cam_x, -cam_y)
            
            # Glow effect
            glow_rect = player_screen_rect.inflate(8, 8)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*self.COLOR_PLAYER_GLOW, 80), glow_surface.get_rect(), border_radius=8)
            self.screen.blit(glow_surface, glow_rect.topleft)

            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_screen_rect, border_radius=4)

    def _render_particles(self):
        cam_x, cam_y = int(self.camera_x), 0
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / p["max_life"]))))
            color = (*p["color"], alpha)
            size = int(p["size"] * (p["life"] / p["max_life"]))
            if size > 0:
                pos = (int(p["pos"][0] - cam_x), int(p["pos"][1] - cam_y))
                # Use gfxdraw for anti-aliased circles if available, else rect
                try:
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
                except:
                    pygame.draw.rect(self.screen, p["color"], (pos[0]-size, pos[1]-size, size*2, size*2))


    def _render_ui(self):
        # Boosts
        boost_text = self.font_ui.render(f"BOOSTS: {self.player['boosts']}", True, self.COLOR_TEXT)
        self.screen.blit(boost_text, (10, 10))

        # Timer
        time_str = f"TIME: {max(0, self.timer // self.FPS):02d}"
        time_text = self.font_ui.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))
        
        # Game Over / Win Message
        if self.game_over:
            msg_str = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_BOOST_PICKUP if self.win else self.COLOR_OBSTACLE
            msg_text = self.font_msg.render(msg_str, True, color)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

    def _generate_level(self):
        self.platforms = []
        self.obstacles = []
        self.boost_pickups = []
        
        # Start platform
        x, y = 100, self.SCREEN_HEIGHT - 50
        self.platforms.append(pygame.Rect(x, y, 200, 50))
        
        level_length = 50
        for i in range(level_length):
            w = self.np_random.integers(100, 250)
            h = self.np_random.integers(40, 80)
            
            gap_x = self.np_random.integers(80, 120) # Horizontal gap
            gap_y = self.np_random.integers(-100, 100) # Vertical gap
            
            x = self.platforms[-1].right + gap_x
            y = np.clip(self.platforms[-1].y + gap_y, 150, self.SCREEN_HEIGHT - h)
            
            self.platforms.append(pygame.Rect(x, y, w, h))

            # Add boost pickups
            if self.np_random.random() < 0.2:
                bx = x + w / 2
                by = y - 40
                self.boost_pickups.append(pygame.Rect(bx - 10, by - 10, 20, 20))

            # Add obstacles
            if self.np_random.random() < 0.3 and i > 2:
                obs_x = x + w / 2
                obs_y = y - 80
                speed = self.np_random.uniform(1, 3) * self.difficulty_modifier
                self.obstacles.append({
                    "rect": pygame.Rect(obs_x - 10, obs_y - 10, 20, 20),
                    "vx": speed, "vy": 0,
                    "min_x": x, "max_x": x + w,
                    "min_y": y - 150, "max_y": y - 50
                })

        # End flag
        last_plat = self.platforms[-1]
        self.end_flag = pygame.Rect(last_plat.centerx, last_plat.top - 50, 10, 50)

    def _spawn_particles(self, pos, count, color, life=20, speed=5, size=3, angle_range=(180, 360)):
        for _ in range(count):
            angle = math.radians(self.np_random.uniform(angle_range[0], angle_range[1]))
            p_speed = self.np_random.uniform(1, speed)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * p_speed, math.sin(angle) * p_speed],
                "life": self.np_random.integers(life // 2, life),
                "max_life": life,
                "color": color,
                "size": self.np_random.integers(size-1, size+2)
            })

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part is for human testing and requires a display.
    # It will not run in a headless environment.
    try:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy" # Force headless for init
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Platformer Environment Test")
        is_headless = False
    except pygame.error:
        print("Pygame display could not be initialized. Running in headless mode.")
        is_headless = True

    obs, info = env.reset()
    done = False
    total_reward = 0

    # Key mapping for human play
    key_map = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
    }

    while not done:
        # Default action is no-op
        action = [0, 0, 0] # movement=none, space=released, shift=released
        
        # Collect keyboard inputs for human play
        if not is_headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            
            # Movement
            for key, move_val in key_map.items():
                if keys[key]:
                    action[0] = move_val
                    break # Prioritize one movement key
            
            # Space and Shift
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

        else: # In headless mode, take random actions
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        if not is_headless:
            # Transpose back to (W, H, C) for pygame display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        if done:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Reset for another round
            obs, info = env.reset()
            done = False
            total_reward = 0
            if is_headless: # Run for one episode in headless
                break

    env.close()