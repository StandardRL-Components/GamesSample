
# Generated: 2025-08-28T04:24:33.350227
# Source Brief: brief_02311.md
# Brief Index: 2311

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Hold [SPACE] to charge jump. Use [←][→] to aim. Release [SPACE] to jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade platformer. Hop between procedurally generated platforms, aiming to reach the top before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_TEXT = (255, 255, 255)
    PLATFORM_PALETTE = [
        (255, 87, 87), (87, 255, 150), (87, 150, 255), (255, 255, 87)
    ]
    COLOR_GOAL = (255, 215, 0)
    
    # Physics & Gameplay
    GRAVITY = 0.5
    INITIAL_SCROLL_SPEED = 1.0
    SCROLL_ACCELERATION = 0.001 # Speed increases per frame
    MAX_SCROLL_SPEED = 4.0
    
    PLAYER_SIZE = (12, 12)
    
    # Jump mechanics
    AIM_SPEED = 0.05
    MIN_AIM_ANGLE = -math.pi * 0.4
    MAX_AIM_ANGLE = math.pi * 0.4
    CHARGE_RATE = 2.0
    MAX_CHARGE = 50.0
    JUMP_MIN_POWER = 5.0
    JUMP_MAX_POWER = 15.0

    # Game rules
    MAX_STEPS = 5400 # 180 seconds at 30 FPS
    INITIAL_LIVES = 3
    TARGET_SCORE = 150

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        self.render_mode = render_mode
        self.np_random = None

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.game_won = False
        
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.player_rect = pygame.Rect(0, 0, *self.PLAYER_SIZE)
        
        self.platforms = []
        self.particles = []
        self.stars = []
        
        self.on_platform = False
        self.current_platform_idx = -1
        self.highest_visited_platform_y = self.HEIGHT

        self.is_charging = False
        self.charge_level = 0.0
        self.aim_angle = 0.0
        self.prev_space_held = False

        self.scroll_speed = self.INITIAL_SCROLL_SPEED
        self.goal_platform_spawned = False
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.game_won = False
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - 50], dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        
        self.platforms = []
        start_platform = {
            "rect": pygame.Rect(self.WIDTH / 2 - 50, self.HEIGHT - 40, 100, 15),
            "color": self.PLATFORM_PALETTE[0],
            "visited": True,
            "is_goal": False
        }
        self.platforms.append(start_platform)
        
        for i in range(10):
            self._spawn_platform()

        self.particles = []
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
            for _ in range(100)
        ]
        
        self.on_platform = True
        self.current_platform_idx = 0
        self.highest_visited_platform_y = start_platform["rect"].top

        self.is_charging = False
        self.charge_level = 0.0
        self.aim_angle = 0.0
        self.prev_space_held = False

        self.scroll_speed = self.INITIAL_SCROLL_SPEED
        self.goal_platform_spawned = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0.0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1

        # 1. Handle Actions
        self._handle_actions(action)
        
        # 2. Update Game State
        self._update_player()
        self._update_platforms()
        self._update_particles()

        # 3. Handle Collisions and Game Logic
        self._handle_collisions()
        
        # 4. Calculate Reward
        reward += 0.01 # Survival reward

        # Check for landing on a new, higher platform
        if self.on_platform and not self.platforms[self.current_platform_idx]["visited"]:
            self.platforms[self.current_platform_idx]["visited"] = True
            platform_y = self.platforms[self.current_platform_idx]["rect"].top
            if platform_y < self.highest_visited_platform_y:
                score_gain = int((self.highest_visited_platform_y - platform_y) / 10)
                self.score += score_gain
                reward += score_gain
                self.highest_visited_platform_y = platform_y
                # SFX: New platform!
            
            # Check for win condition
            if self.platforms[self.current_platform_idx]["is_goal"]:
                self.game_won = True
                self.game_over = True
                terminated = True
                reward = 100.0
                # SFX: Victory!
        
        # 5. Check Termination Conditions
        if self.player_pos[1] > self.HEIGHT + self.PLAYER_SIZE[1]:
            # Player fell off screen
            self.lives -= 1
            reward -= 25.0 # Penalty for falling
            # SFX: Fall/Hurt
            if self.lives <= 0:
                self.game_over = True
                terminated = True
                reward = -100.0 # Terminal penalty
                # SFX: Game Over
            else:
                self._respawn_player()

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            if not self.game_won:
                reward = -50.0 # Penalty for running out of time
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_action, _ = action
        space_held = space_action == 1
        is_jump_triggered = not space_held and self.prev_space_held

        if self.on_platform:
            if space_held:
                self.is_charging = True
                self.charge_level = min(self.MAX_CHARGE, self.charge_level + self.CHARGE_RATE)
                if movement == 3: # Left
                    self.aim_angle -= self.AIM_SPEED
                elif movement == 4: # Right
                    self.aim_angle += self.AIM_SPEED
                self.aim_angle = np.clip(self.aim_angle, self.MIN_AIM_ANGLE, self.MAX_AIM_ANGLE)
            elif is_jump_triggered and self.charge_level > 5: # Minimum charge to jump
                # SFX: Jump!
                jump_power = self.JUMP_MIN_POWER + (self.charge_level / self.MAX_CHARGE) * (self.JUMP_MAX_POWER - self.JUMP_MIN_POWER)
                self.player_vel[0] = math.sin(self.aim_angle) * jump_power
                self.player_vel[1] = -math.cos(self.aim_angle) * jump_power
                self.on_platform = False
                self.is_charging = False
                self.current_platform_idx = -1
            else: # Not charging, not jumping
                self.charge_level = max(0, self.charge_level - self.CHARGE_RATE * 4) # Decay charge if not held
                if self.charge_level == 0:
                    self.is_charging = False
        
        self.prev_space_held = space_held

    def _update_player(self):
        if not self.on_platform:
            self.player_vel[1] += self.GRAVITY
            self.player_pos += self.player_vel

            # Wall bouncing
            if self.player_pos[0] < 0 or self.player_pos[0] > self.WIDTH - self.PLAYER_SIZE[0]:
                self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH - self.PLAYER_SIZE[0])
                self.player_vel[0] *= -0.5 # Lose some horizontal energy on bounce

        self.player_rect.topleft = self.player_pos

    def _update_platforms(self):
        self.scroll_speed = min(self.MAX_SCROLL_SPEED, self.scroll_speed + self.SCROLL_ACCELERATION)
        
        for p in self.platforms:
            p["rect"].y += self.scroll_speed

        # Prune platforms that have scrolled off screen
        self.platforms = [p for p in self.platforms if p["rect"].top < self.HEIGHT]
        
        # Spawn new platforms if needed
        if self.platforms:
            highest_p_y = min(p["rect"].y for p in self.platforms)
            if highest_p_y > -100:
                self._spawn_platform()

    def _spawn_platform(self):
        last_platform = max(self.platforms, key=lambda p: p["rect"].y if p["rect"].y < 0 else -float('inf'))
        last_rect = last_platform["rect"]

        is_goal = self.score >= self.TARGET_SCORE and not self.goal_platform_spawned
        
        if is_goal:
            width = 150
            height = 20
            color = self.COLOR_GOAL
            self.goal_platform_spawned = True
        else:
            width = self.np_random.integers(60, 120)
            height = 15
            color = self.np_random.choice(self.PLATFORM_PALETTE)

        dx = self.np_random.integers(-180, 180)
        dy = self.np_random.integers(80, 130)

        new_x = last_rect.centerx + dx
        new_x = np.clip(new_x - width / 2, 20, self.WIDTH - width - 20)
        new_y = last_rect.top - dy
        
        new_platform = {
            "rect": pygame.Rect(int(new_x), int(new_y), width, height),
            "color": color,
            "visited": False,
            "is_goal": is_goal
        }
        self.platforms.append(new_platform)
    
    def _handle_collisions(self):
        if self.player_vel[1] > 0: # Only check for landing if falling
            for i, p in enumerate(self.platforms):
                if self.player_rect.colliderect(p["rect"]):
                    # Check if player was above platform in previous frame
                    if self.player_pos[1] + self.PLAYER_SIZE[1] - self.player_vel[1] <= p["rect"].top:
                        # SFX: Land!
                        self.on_platform = True
                        self.player_vel = np.zeros(2, dtype=np.float32)
                        self.player_pos[1] = p["rect"].top - self.PLAYER_SIZE[1]
                        self.player_rect.topleft = self.player_pos
                        self.charge_level = 0.0
                        self.aim_angle = 0.0
                        self.current_platform_idx = i
                        self._create_landing_particles(self.player_rect.midbottom, p["color"])
                        return

    def _respawn_player(self):
        # Find highest visited platform to respawn on
        respawn_platform = self.platforms[0]
        highest_y = self.HEIGHT
        for p in self.platforms:
            if p["visited"] and p["rect"].top < highest_y:
                highest_y = p["rect"].top
                respawn_platform = p
        
        self.player_pos = np.array([respawn_platform["rect"].centerx, respawn_platform["rect"].top - self.PLAYER_SIZE[1]], dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.on_platform = True
        self.is_charging = False
        self.charge_level = 0.0
        self.aim_angle = 0.0
        self.current_platform_idx = self.platforms.index(respawn_platform)

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "time_left": (self.MAX_STEPS - self.steps) // self.FPS,
        }
        
    def _render_background(self):
        # Gradient background
        for y in range(self.HEIGHT):
            color = [
                int(self.COLOR_BG_TOP[i] + (self.COLOR_BG_BOTTOM[i] - self.COLOR_BG_TOP[i]) * (y / self.HEIGHT))
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Stars
        for x, y, size in self.stars:
            c = self.np_random.integers(50, 150)
            pygame.draw.rect(self.screen, (c,c,c), (x, (y + int(self.steps * 0.1 * size)) % self.HEIGHT, size, size))

    def _render_game(self):
        for p in self.platforms:
            pygame.draw.rect(self.screen, p["color"], p["rect"], border_radius=3)
            if p["is_goal"]:
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, p["rect"].inflate(-8, -8), 2, border_radius=3)

        self._render_particles()

        # Render player
        glow_size = int(self.PLAYER_SIZE[0] * (1 + self.charge_level / self.MAX_CHARGE))
        glow_alpha = int(50 + 100 * (self.charge_level / self.MAX_CHARGE))
        pygame.gfxdraw.filled_circle(self.screen, self.player_rect.centerx, self.player_rect.centery, glow_size, (*self.COLOR_PLAYER_GLOW, glow_alpha))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=2)

        # Render jump trajectory
        if self.is_charging:
            self._render_trajectory()

    def _render_ui(self):
        time_left = max(0, (self.MAX_STEPS - self.steps) // self.FPS)
        time_text = self.font_small.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 10))

        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH / 2 - lives_text.get_width() / 2, 10))

        if self.game_over:
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            end_text = self.font_large.render(message, True, self.COLOR_TEXT)
            self.screen.blit(end_text, (self.WIDTH / 2 - end_text.get_width() / 2, self.HEIGHT / 2 - end_text.get_height() / 2))

    def _render_trajectory(self):
        jump_power = self.JUMP_MIN_POWER + (self.charge_level / self.MAX_CHARGE) * (self.JUMP_MAX_POWER - self.JUMP_MIN_POWER)
        vx = math.sin(self.aim_angle) * jump_power
        vy = -math.cos(self.aim_angle) * jump_power
        
        path_points = []
        px, py = self.player_rect.center
        for t in range(1, 15):
            new_px = px + vx * t
            new_py = py + vy * t + 0.5 * self.GRAVITY * t * t
            path_points.append((new_px, new_py))
        
        if len(path_points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_PLAYER, False, path_points)
            
    def _create_landing_particles(self, pos, color):
        for _ in range(15):
            vel = [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -1)]
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.2 # Particle gravity
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
    
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 20))
            color = (*p["color"], alpha)
            size = max(1, int(p["lifespan"] / 4))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p["pos"][0] - size), int(p["pos"][1] - size)), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # Example of how to run the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Use this to test the game with keyboard controls
    import sys
    
    pygame.display.set_caption("Icy Ascent")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    running = True
    total_reward = 0
    
    # Map keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while running:
        # Get keyboard input
        keys = pygame.key.get_pressed()
        
        movement_action = 0 # No-op
        for key, action_val in key_map.items():
            if keys[key]:
                movement_action = action_val
                break
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            print("Press 'R' to reset.")

    env.close()
    pygame.quit()
    sys.exit()