
# Generated: 2025-08-27T13:44:10.394456
# Source Brief: brief_00468.md
# Brief Index: 468

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: The character runs automatically. Press SPACE to jump and SHIFT to dash."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced side-view runner. Precisely time jumps and dashes to overcome procedurally generated obstacles and reach the end of the level within the time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -15
        self.PLAYER_SPEED = 6
        self.DASH_SPEED_BONUS = 12
        self.DASH_DURATION = 8  # frames
        self.LEVEL_LENGTH = self.SCREEN_WIDTH * 15 # Approx 15 screens long
        self.TIME_LIMIT_SECONDS = 30
        self.MAX_EPISODE_STEPS = 1000

        # Colors
        self.COLOR_BG_TOP = (40, 60, 110)
        self.COLOR_BG_BOTTOM = (80, 100, 160)
        self.COLOR_GROUND = (87, 56, 42)
        self.COLOR_PLAYER = (255, 220, 0)
        self.COLOR_PLAYER_DASH = (255, 255, 255)
        self.COLOR_OBSTACLE = (210, 50, 50)
        self.COLOR_OBSTACLE_BREAKABLE = (255, 120, 120)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (255, 255, 255)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.state = {} # Will be populated by reset
        
        # This is a dummy call to initialize state before validation
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.state = {
            "steps": 0,
            "score": 0.0,
            "terminated": False,
            "win": False,
            "time_remaining": self.TIME_LIMIT_SECONDS * self.FPS,
            
            "player_pos": np.array([100.0, self.SCREEN_HEIGHT - 80.0]),
            "player_vel": np.array([0.0, 0.0]),
            "on_ground": True,
            "is_dashing": False,
            "dash_timer": 0,
            
            "level_progress": 0.0,
            "camera_x": 0.0,
            
            "obstacles": [],
            "particles": [],
            
            "next_obstacle_x": self.SCREEN_WIDTH,
            "base_obstacle_freq": 180,
            "base_gap_freq": 0.08,
            "obstacle_speed_multiplier": 1.0,
            
            "prev_space_held": False,
            "prev_shift_held": False,
            "pending_reward": 0.0,
        }
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.state["terminated"]:
            # If the game is over, do nothing until reset
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # Reset pending reward
        self.state["pending_reward"] = 0.0

        # Update game logic
        self._update_player(space_held, shift_held)
        self._update_world()
        self._handle_collisions()
        self._update_particles()
        
        self.state["steps"] += 1
        self.state["time_remaining"] -= 1
        
        # Calculate step reward
        reward = 0.1 + self.state["pending_reward"] # Survival bonus + event rewards

        # Check for termination conditions
        self._check_termination()
        
        if self.state["terminated"]:
            if self.state["win"]:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Fail penalty
        
        self.state["score"] += reward
        
        return (
            self._get_observation(),
            reward,
            self.state["terminated"],
            False,
            self._get_info()
        )

    def _update_player(self, space_held, shift_held):
        s = self.state

        # Handle Jump (on rising edge)
        if space_held and not s["prev_space_held"] and s["on_ground"]:
            s["player_vel"][1] = self.JUMP_STRENGTH
            s["on_ground"] = False
            # sfx: jump

        # Handle Dash (on rising edge)
        if shift_held and not s["prev_shift_held"] and not s["is_dashing"]:
            s["is_dashing"] = True
            s["dash_timer"] = self.DASH_DURATION
            # sfx: dash
            for _ in range(20):
                self._create_particle(s["player_pos"].copy(), is_dash=True)

        s["prev_space_held"] = space_held
        s["prev_shift_held"] = shift_held

        # Update dash state
        if s["is_dashing"]:
            s["dash_timer"] -= 1
            if s["dash_timer"] <= 0:
                s["is_dashing"] = False

        # Apply gravity
        s["player_vel"][1] += self.GRAVITY
        
        # Update position
        s["player_pos"] += s["player_vel"]

        # Ground collision
        ground_y = self.SCREEN_HEIGHT - 80
        if s["player_pos"][1] >= ground_y:
            s["player_pos"][1] = ground_y
            s["player_vel"][1] = 0
            if not s["on_ground"]:
                # sfx: land
                for _ in range(5):
                    self._create_particle(s["player_pos"] + np.array([0, 20]), is_dash=False)
            s["on_ground"] = True

        # Keep player within horizontal bounds
        s["player_pos"][0] = np.clip(s["player_pos"][0], 0, self.SCREEN_WIDTH - 30)

    def _update_world(self):
        s = self.state
        
        # Player horizontal movement
        current_speed = self.PLAYER_SPEED
        if s["is_dashing"]:
            current_speed += self.DASH_SPEED_BONUS
        
        s["level_progress"] += current_speed
        
        # Update camera
        s["camera_x"] = s["level_progress"] - 100

        # Update difficulty
        difficulty_tier = s["steps"] // 250
        s["obstacle_speed_multiplier"] = 1.0 + 0.05 * difficulty_tier
        current_gap_freq = self.state["base_gap_freq"] + 0.02 * difficulty_tier
        current_obstacle_freq = self.state["base_obstacle_freq"] / (1 + 0.05 * difficulty_tier)

        # Procedurally generate obstacles and gaps
        while s["next_obstacle_x"] < s["camera_x"] + self.SCREEN_WIDTH + 100:
            is_gap = self.np_random.random() < current_gap_freq
            if is_gap:
                gap_width = self.np_random.integers(100, 200)
                s["obstacles"].append({"type": "gap", "x": s["next_obstacle_x"], "width": gap_width})
                s["next_obstacle_x"] += gap_width + self.np_random.integers(150, 250)
            else:
                is_breakable = self.np_random.random() < 0.3
                obstacle_height = self.np_random.integers(40, 80)
                s["obstacles"].append({
                    "type": "breakable" if is_breakable else "solid",
                    "x": s["next_obstacle_x"],
                    "y": self.SCREEN_HEIGHT - 80 - obstacle_height,
                    "width": 30,
                    "height": obstacle_height,
                    "rect": pygame.Rect(0,0,0,0) # pre-alloc
                })
                s["next_obstacle_x"] += self.np_random.integers(int(current_obstacle_freq), int(current_obstacle_freq * 2))

        # Remove off-screen obstacles
        s["obstacles"] = [obs for obs in s["obstacles"] if obs["x"] + obs.get("width", 30) > s["camera_x"]]

    def _handle_collisions(self):
        s = self.state
        player_rect = pygame.Rect(int(s["player_pos"][0]), int(s["player_pos"][1]), 30, 40)
        
        on_solid_ground = True
        for obs in s["obstacles"]:
            if obs["type"] == "gap":
                if obs["x"] < s["player_pos"][0] + 15 < obs["x"] + obs["width"]:
                    on_solid_ground = False
                continue

            obs_rect = pygame.Rect(
                int(obs["x"] - s["camera_x"]),
                int(obs["y"]),
                obs["width"],
                obs["height"]
            )
            
            if player_rect.colliderect(obs_rect):
                if obs["type"] == "breakable" and s["is_dashing"]:
                    # sfx: break_obstacle
                    s["pending_reward"] += 1.0
                    for _ in range(15):
                        self._create_particle(np.array([obs_rect.centerx, obs_rect.centery]), is_dash=False)
                    s["obstacles"].remove(obs)
                else:
                    # sfx: hit_obstacle
                    s["pending_reward"] -= 5.0
                    s["terminated"] = True
                    for _ in range(30):
                        self._create_particle(s["player_pos"] + np.array([15, 20]), is_dash=False)
                    return
        
        if not on_solid_ground and s["on_ground"]:
            # sfx: fall
            s["terminated"] = True

    def _check_termination(self):
        s = self.state
        if s["terminated"]: # Already terminated by collision
            return
        
        if s["level_progress"] >= self.LEVEL_LENGTH:
            s["terminated"] = True
            s["win"] = True
        elif s["time_remaining"] <= 0:
            s["terminated"] = True
        elif s["steps"] >= self.MAX_EPISODE_STEPS:
            s["terminated"] = True
        elif s["player_pos"][1] > self.SCREEN_HEIGHT: # Fell off screen
             s["terminated"] = True

    def _update_particles(self):
        s = self.state
        for p in s["particles"]:
            p["pos"] += p["vel"]
            p["vel"][1] += 0.2 # particle gravity
            p["life"] -= 1
        s["particles"] = [p for p in s["particles"] if p["life"] > 0]

    def _create_particle(self, pos, is_dash):
        angle = self.np_random.random() * 2 * math.pi
        speed = self.np_random.random() * (5 if is_dash else 3) + 2
        vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        if is_dash:
            vel[0] -= self.PLAYER_SPEED # Trail effect
        self.state["particles"].append({
            "pos": pos,
            "vel": vel,
            "life": self.np_random.integers(10, 25),
            "size": self.np_random.integers(2, 5)
        })

    def _get_observation(self):
        # Clear screen with background gradient
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        s = self.state
        cam_x = int(s["camera_x"])

        # Draw ground
        last_x = 0
        ground_y = self.SCREEN_HEIGHT - 80
        for obs in sorted([o for o in s["obstacles"] if o['type'] == 'gap'], key=lambda o: o['x']):
            pygame.draw.rect(self.screen, self.COLOR_GROUND, (last_x - cam_x, ground_y, obs['x'] - last_x, 80))
            last_x = obs['x'] + obs['width']
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (last_x - cam_x, ground_y, self.SCREEN_WIDTH - (last_x-cam_x) + 100, 80))

        # Draw obstacles
        for obs in s["obstacles"]:
            if obs["type"] == "gap": continue
            color = self.COLOR_OBSTACLE_BREAKABLE if obs["type"] == "breakable" else self.COLOR_OBSTACLE
            rect = pygame.Rect(int(obs["x"] - cam_x), int(obs["y"]), obs["width"], obs["height"])
            pygame.draw.rect(self.screen, color, rect)

        # Draw particles
        for p in s["particles"]:
            pos = p["pos"].copy()
            pos[0] -= cam_x
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, pos.astype(int), int(p["size"] * (p["life"] / 25.0)))

        # Draw player
        player_color = self.COLOR_PLAYER_DASH if s["is_dashing"] else self.COLOR_PLAYER
        player_rect = pygame.Rect(int(s["player_pos"][0]), int(s["player_pos"][1]), 30, 40)
        
        # Simple animation
        if not s["on_ground"]: # Jump/Fall pose
            player_rect.height = 45
            player_rect.width = 25
            player_rect.x += 2
        elif s["is_dashing"]: # Dash pose
             player_rect.height = 20
             player_rect.y += 20
             player_rect.width = 45
        else: # Running animation
            bob = math.sin(s["steps"] * 0.5) * 3
            player_rect.y += int(bob)
        
        pygame.draw.rect(self.screen, player_color, player_rect)
        pygame.gfxdraw.rectangle(self.screen, player_rect, (0,0,0)) # Outline

    def _render_ui(self):
        s = self.state
        
        # Timer
        time_text = f"TIME: {s['time_remaining'] / self.FPS:.1f}"
        text_surface = self.font_large.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - text_surface.get_width() - 20, 10))
        
        # Progress Bar
        progress_ratio = min(1.0, s["level_progress"] / self.LEVEL_LENGTH)
        bar_width = self.SCREEN_WIDTH - 40
        bar_height = 10
        pygame.draw.rect(self.screen, (50,50,50), (20, self.SCREEN_HEIGHT - 30, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (20, self.SCREEN_HEIGHT - 30, bar_width * progress_ratio, bar_height))

        # Game Over message
        if s["terminated"]:
            msg = "LEVEL COMPLETE!" if s["win"] else "GAME OVER"
            color = (100, 255, 100) if s["win"] else (255, 100, 100)
            msg_surface = self.font_large.render(msg, True, color)
            msg_rect = msg_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surface, msg_rect)

    def _get_info(self):
        return {
            "score": self.state["score"],
            "steps": self.state["steps"],
            "level_progress": self.state["level_progress"],
            "time_remaining": self.state["time_remaining"]
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example usage:
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # To run the game with manual controls:
    import pygame
    from pygame.locals import K_SPACE, K_LSHIFT, K_r
    
    obs, info = env.reset()
    terminated = False
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Runner")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    running = True
    while running:
        space_pressed = False
        shift_pressed = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False

        keys = pygame.key.get_pressed()
        if keys[K_SPACE]:
            space_pressed = True
        if keys[K_LSHIFT]:
            shift_pressed = True

        action = [0, 1 if space_pressed else 0, 1 if shift_pressed else 0]
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # The environment returns an observation with the correct channel order for numpy,
        # but pygame.surfarray.make_surface expects a transposed version.
        # So we transpose it back for rendering.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()