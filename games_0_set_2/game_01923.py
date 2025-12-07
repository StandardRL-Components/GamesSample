
# Generated: 2025-08-28T03:07:43.089161
# Source Brief: brief_01923.md
# Brief Index: 1923

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Use ←→ to aim your jump. Use 'Up Arrow' for a small jump, "
        "'Shift' for a medium jump, and 'Space' for a large jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced procedural platformer. Jump between scrolling platforms to reach the "
        "goal flag before time runs out. Higher jumps are riskier but cover more distance."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.screen_width = 640
        self.screen_height = 400
        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)

        # Game constants
        self.max_steps = 1800  # 60 seconds at 30 FPS
        self.gravity = 0.4
        self.scroll_speed = 1.0
        self.player_size = 20

        # Colors
        self.COLOR_BG_TOP = (20, 30, 80)
        self.COLOR_BG_BOTTOM = (60, 80, 150)
        self.COLOR_PLATFORM = (120, 120, 120)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 150)
        self.COLOR_FLAGPOLE = (100, 200, 100)
        self.COLOR_FLAG = (255, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (255, 255, 220)
        
        # State variables will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.platforms = []
        self.flag_pole = None
        self.flag = None
        self.particles = []
        self.highest_platform_idx = 0
        self.fall_penalty_applied = False
        
        # Difficulty scaling
        self.base_min_v_gap = 60
        self.base_max_v_gap = 120
        self.base_h_gap_var = 50
        self.min_v_gap = self.base_min_v_gap
        self.max_v_gap = self.base_max_v_gap
        self.h_gap_var = self.base_h_gap_var

        # Initialize state variables
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.fall_penalty_applied = False
        
        # Reset difficulty
        self.min_v_gap = self.base_min_v_gap
        self.max_v_gap = self.base_max_v_gap
        self.h_gap_var = self.base_h_gap_var

        # Player state
        self.player = {
            "rect": pygame.Rect(self.screen_width // 2 - self.player_size // 2, self.screen_height - 100, self.player_size, self.player_size),
            "vx": 0,
            "vy": 0,
            "on_ground": True
        }

        # Procedural generation
        self.platforms = []
        start_platform = pygame.Rect(self.player['rect'].x - 40, self.player['rect'].y + self.player_size, 100, 20)
        self.platforms.append(start_platform)
        
        self.highest_platform_idx = 0

        # Generate initial platforms
        while len(self.platforms) < 15:
            self._add_platform()

        # Place flag on the highest platform
        top_platform = min(self.platforms, key=lambda p: p.y)
        self.flag_pole = pygame.Rect(top_platform.centerx - 2, top_platform.y - 50, 4, 50)
        self.flag = [
            (self.flag_pole.x, self.flag_pole.y),
            (self.flag_pole.x, self.flag_pole.y + 20),
            (self.flag_pole.x - 25, self.flag_pole.y + 10)
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Per-step penalty
        self.steps += 1

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if self.player["on_ground"]:
            jump_triggered = False
            jump_vy = 0
            jump_vx = 0
            
            # Determine jump power
            if space_held: # Large Jump
                jump_vy = -11
                jump_triggered = True
            elif shift_held: # Medium Jump
                jump_vy = -8.5
                jump_triggered = True
            elif movement == 1: # Small Jump (Up Arrow)
                jump_vy = -6
                jump_triggered = True

            if jump_triggered:
                # Determine horizontal direction
                if movement == 3: # Left
                    jump_vx = -4
                elif movement == 4: # Right
                    jump_vx = 4
                
                self.player["vy"] = jump_vy
                self.player["vx"] = jump_vx
                self.player["on_ground"] = False
                # sfx: jump
                self._create_particles(self.player["rect"].midbottom, 5, "jump")

        # --- Physics and World Update ---
        # Apply gravity
        if not self.player["on_ground"]:
            self.player["vy"] += self.gravity

        # Update player position
        self.player["rect"].x += int(self.player["vx"])
        self.player["rect"].y += int(self.player["vy"])

        # World scroll
        self.player["rect"].y += self.scroll_speed
        for p in self.platforms:
            p.y += self.scroll_speed
        self.flag_pole.y += self.scroll_speed
        self.flag = [(x, y + self.scroll_speed) for x, y in self.flag]
        
        # --- Collision Detection ---
        self.player["on_ground"] = False
        player_bottom = self.player["rect"].bottom
        
        for i, plat in enumerate(self.platforms):
            if (self.player["rect"].colliderect(plat) and 
                self.player["vy"] >= 0 and 
                player_bottom - self.player["vy"] <= plat.top + 1): # Check if player was above last frame
                
                self.player["rect"].bottom = plat.top
                self.player["vy"] = 0
                self.player["vx"] *= 0.5 # Ground friction
                self.player["on_ground"] = True
                self.fall_penalty_applied = False # Reset fall penalty flag
                
                reward += 0.1 # Reward for landing
                # sfx: land
                self._create_particles(self.player["rect"].midbottom, 3, "land")

                # Check for new height
                if i < self.highest_platform_idx: # Lower index means higher platform
                    reward += 1.0
                    self.score += 10
                    self.highest_platform_idx = i
                break # Player can only be on one platform

        # --- Game State Update ---
        # Remove off-screen platforms and add new ones
        self.platforms = [p for p in self.platforms if p.top < self.screen_height]
        while len(self.platforms) < 15:
            self._add_platform()
            
        # Update difficulty
        if self.steps > 0 and self.steps % 50 == 0:
            self.max_v_gap += 0.5
        if self.steps > 0 and self.steps % 100 == 0:
            self.h_gap_var += 1

        # Keep player on screen horizontally
        if self.player["rect"].left < 0:
            self.player["rect"].left = 0
            self.player["vx"] = 0
        if self.player["rect"].right > self.screen_width:
            self.player["rect"].right = self.screen_width
            self.player["vx"] = 0
        
        # --- Termination and Final Rewards ---
        terminated = False
        highest_y_pos = self.platforms[self.highest_platform_idx].y
        
        # 1. Fall penalty
        if not self.fall_penalty_applied and self.player["rect"].top > highest_y_pos + self.screen_height / 2:
            reward -= 1.0
            self.fall_penalty_applied = True
            
        # 2. Fell off screen
        if self.player["rect"].top > self.screen_height:
            terminated = True
            self.game_over = True
            reward = -10.0
            self.score -= 50
            # sfx: fail

        # 3. Reached the flag
        player_center = self.player["rect"].center
        if self.flag_pole.collidepoint(player_center):
            terminated = True
            self.game_over = True
            reward = 100.0
            self.score += 1000
            # sfx: win

        # 4. Time ran out
        if self.steps >= self.max_steps:
            terminated = True
            self.game_over = True
            # No specific reward change, just ends.

        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _add_platform(self):
        last_platform = min(self.platforms, key=lambda p: p.y)
        
        gap_y = self.np_random.uniform(self.min_v_gap, self.max_v_gap)
        new_y = last_platform.y - gap_y
        
        max_offset = self.h_gap_var
        offset_x = self.np_random.uniform(-max_offset, max_offset)
        new_x = last_platform.centerx + offset_x
        
        width = self.np_random.uniform(80, 150)
        
        # Clamp to screen bounds
        new_x = np.clip(new_x, width / 2, self.screen_width - width / 2)
        
        new_platform = pygame.Rect(int(new_x - width / 2), int(new_y), int(width), 20)
        self.platforms.append(new_platform)

    def _create_particles(self, pos, count, p_type):
        for _ in range(count):
            if p_type == "jump":
                vel = (self.np_random.uniform(-1, 1), self.np_random.uniform(0.5, 2))
            else: # land
                vel = (self.np_random.uniform(-1.5, 1.5), self.np_random.uniform(-2, -0.5))
            
            particle = {
                "pos": list(pos),
                "vel": vel,
                "life": self.np_random.integers(10, 20),
                "size": self.np_random.uniform(3, 6)
            }
            self.particles.append(particle)

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["size"] -= 0.2
            p["life"] -= 1
            if p["life"] <= 0 or p["size"] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p["life"] / 20))))
                temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, self.COLOR_PARTICLE + (alpha,), (int(p["size"]), int(p["size"])), int(p["size"]))
                self.screen.blit(temp_surf, (int(p["pos"][0] - p["size"]), int(p["pos"][1] - p["size"])))

    def _get_observation(self):
        # --- Draw Background Gradient ---
        for y in range(self.screen_height):
            interp = y / self.screen_height
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))

        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)
            pygame.draw.rect(self.screen, (0,0,0), plat, width=1, border_radius=3)

        # Draw flag
        pygame.draw.rect(self.screen, self.COLOR_FLAGPOLE, self.flag_pole)
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, self.flag)
        pygame.draw.aalines(self.screen, (0,0,0), True, self.flag)

        # Draw particles
        self._update_and_draw_particles()

        # Draw player with glow
        glow_size = self.player_size * 1.5
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW + (100,), (glow_size/2, glow_size/2), glow_size/2)
        self.screen.blit(glow_surf, (self.player["rect"].centerx - glow_size/2, self.player["rect"].centery - glow_size/2))
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player["rect"], border_radius=3)
        pygame.draw.rect(self.screen, (50,50,0), self.player["rect"], width=2, border_radius=3)


    def _render_ui(self):
        # Timer
        seconds_left = max(0, (self.max_steps - self.steps) // 30)
        timer_text = f"TIME: {seconds_left:02d}"
        text_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.screen_width - text_surf.get_width() - 10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment for human play
if __name__ == '__main__':
    import os
    # Set SDL to a dummy driver to run headless
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    # Set render_mode to "human" to visualize, or "rgb_array" for headless training
    render_mode = "human"
    
    if render_mode == "human":
        # For human playback, we need a display
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
        pygame.display.set_caption("Procedural Platformer")
        screen = pygame.display.set_mode((640, 400))
    
    env = GameEnv(render_mode=render_mode)
    
    # Validation check
    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env.unwrapped)
        print("✓ Gymnasium environment check passed.")
    except Exception as e:
        print(f"✗ Gymnasium environment check failed: {e}")

    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Use a basic controller for human play
    keys_pressed = {
        "up": False, "down": False, "left": False, "right": False,
        "space": False, "shift": False
    }

    print("\n" + env.game_description)
    print(env.user_guide)
    print("Press 'R' to reset the game.")

    while running:
        action = [0, 0, 0] # Default no-op
        
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: keys_pressed["up"] = True
                    if event.key == pygame.K_DOWN: keys_pressed["down"] = True
                    if event.key == pygame.K_LEFT: keys_pressed["left"] = True
                    if event.key == pygame.K_RIGHT: keys_pressed["right"] = True
                    if event.key == pygame.K_SPACE: keys_pressed["space"] = True
                    if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_pressed["shift"] = True
                    if event.key == pygame.K_r: # Reset on 'r'
                        obs, info = env.reset()
                        total_reward = 0
                        print("--- Game Reset ---")
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_UP: keys_pressed["up"] = False
                    if event.key == pygame.K_DOWN: keys_pressed["down"] = False
                    if event.key == pygame.K_LEFT: keys_pressed["left"] = False
                    if event.key == pygame.K_RIGHT: keys_pressed["right"] = False
                    if event.key == pygame.K_SPACE: keys_pressed["space"] = False
                    if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_pressed["shift"] = False

            # Map keys to MultiDiscrete action
            if keys_pressed["up"]: action[0] = 1
            elif keys_pressed["down"]: action[0] = 2 # Note: 'down' has no jump effect
            elif keys_pressed["left"]: action[0] = 3
            elif keys_pressed["right"]: action[0] = 4
            else: action[0] = 0
            
            action[1] = 1 if keys_pressed["space"] else 0
            action[2] = 1 if keys_pressed["shift"] else 0
        else:
            # For headless mode, just sample actions
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render_mode == "human":
            # The observation is already a rendered frame, so we just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30) # Control the frame rate

        if terminated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Score: {info['score']:.0f}, Steps: {info['steps']}")
            total_reward = 0
            if render_mode != "human": # In headless, run one episode and stop
                running = False
            else: # In human mode, wait for reset key
                pass
                
    env.close()