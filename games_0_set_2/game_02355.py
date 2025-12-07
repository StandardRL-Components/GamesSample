
# Generated: 2025-08-28T04:33:28.544586
# Source Brief: brief_02355.md
# Brief Index: 2355

        
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

    # User-facing strings for the Haunted House game
    user_guide = (
        "Controls: ←→ to run, ↑ to jump. Press ↓ in dark blue hiding spots to hide from ghosts."
    )
    game_description = (
        "Escape a procedurally generated haunted house by running, jumping, and hiding from ghosts before time runs out."
    )

    # Frames auto-advance at 30fps
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.W, self.H = 640, 400
        self.FPS = 30
        self.NUM_STAGES = 3
        self.ROOMS_PER_STAGE = 5
        self.WORLD_W = self.W * self.ROOMS_PER_STAGE

        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_PLATFORM = (40, 45, 60)
        self.COLOR_HIDING_SPOT = (25, 30, 80)
        self.COLOR_HIDING_SPOT_LINE = (35, 40, 100)
        self.COLOR_EXIT = (40, 160, 100)
        self.COLOR_PLAYER = (60, 220, 255)
        self.COLOR_GHOST = (255, 50, 80)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TIMER_WARN = (255, 200, 0)
        self.COLOR_TIMER_DANGER = (255, 80, 80)

        # Physics & Player
        self.GRAVITY = 0.8
        self.PLAYER_SPEED = 6
        self.JUMP_POWER = 14
        self.PLAYER_W, self.PLAYER_H = 20, 32

        # Ghost
        self.GHOST_SPEED_BASE = 1.5
        self.GHOST_ALERT_RADIUS = 250
        self.GHOST_W, self.GHOST_H = 28, 40

        # Game State
        self.max_steps = 180 * self.FPS  # 180 seconds total
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # --- Initialize State Variables ---
        # These will be properly set in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stage = 1
        self.time_left = 0
        self.np_random = None
        
        self.player = {}
        self.ghosts = []
        self.platforms = []
        self.hiding_spots = []
        self.exit_door = None
        self.particles = []
        
        self.camera_x = 0
        self.last_room_idx = 0
        self.flicker_alpha = 0

        self.validate_implementation()
    
    def _generate_stage(self):
        self.platforms = []
        self.hiding_spots = []
        
        # Create continuous floor for the stage
        self.platforms.append(pygame.Rect(0, self.H - 20, self.WORLD_W, 20))

        # Procedurally generate platforms and hiding spots for each room in the stage
        for room_idx in range(self.ROOMS_PER_STAGE):
            room_x_start = room_idx * self.W
            
            # Don't put obstacles in the first room of the first stage
            if self.stage == 1 and room_idx == 0:
                continue

            # Add platforms
            num_platforms = self.np_random.integers(self.stage, self.stage + 3)
            for _ in range(num_platforms):
                w = self.np_random.integers(80, 250)
                h = 20
                px = room_x_start + self.np_random.integers(0, self.W - w)
                py = self.np_random.integers(100, self.H - 80)
                # Ensure platforms don't overlap too much vertically
                py = (py // 60) * 60 + 30
                self.platforms.append(pygame.Rect(px, py, w, h))

            # Add hiding spots
            num_hiding_spots = self.np_random.integers(1, 3)
            for _ in range(num_hiding_spots):
                w = self.np_random.integers(40, 70)
                h = self.np_random.integers(60, 120)
                px = room_x_start + self.np_random.integers(0, self.W - w)
                # Place them on the floor or on a platform
                potential_surfaces = [p for p in self.platforms if p.left < px + w and p.right > px]
                if potential_surfaces:
                    # Choose the lowest platform to place the object on
                    surface = min(potential_surfaces, key=lambda r: r.top)
                    py = surface.top - h
                    self.hiding_spots.append(pygame.Rect(px, py, w, h))
        
        # Place exit door at the end of the stage
        exit_w, exit_h = 40, 80
        self.exit_door = pygame.Rect(self.WORLD_W - exit_w - 20, self.H - 20 - exit_h, exit_w, exit_h)

    def _reset_player_and_ghosts(self):
        # Player
        self.player = {
            "rect": pygame.Rect(100, self.H - 40 - self.PLAYER_H, self.PLAYER_W, self.PLAYER_H),
            "vx": 0, "vy": 0,
            "on_ground": False,
            "is_hiding": False
        }
        
        # Ghosts
        self.ghosts = []
        num_ghosts = 3
        ghost_speed_multiplier = 1 + 0.5 * (self.stage - 1)
        
        # Ghosts start in rooms 1, 2, 3 (0-indexed)
        possible_rooms = list(range(1, self.ROOMS_PER_STAGE))
        start_rooms = self.np_random.choice(possible_rooms, size=min(num_ghosts, len(possible_rooms)), replace=False)
        
        for i in range(num_ghosts):
            home_room_idx = start_rooms[i % len(start_rooms)]
            self.ghosts.append({
                "rect": pygame.Rect(home_room_idx * self.W + self.W/2, self.H/2, self.GHOST_W, self.GHOST_H),
                "state": "patrol", # patrol, chase
                "target": None,
                "home_room_idx": home_room_idx,
                "speed": self.GHOST_SPEED_BASE * ghost_speed_multiplier,
                "bob_phase": self.np_random.uniform(0, 2 * math.pi)
            })

    def _advance_stage(self):
        self.stage += 1
        if self.stage > self.NUM_STAGES:
            self.win = True
            self.game_over = True
        else:
            self._generate_stage()
            self._reset_player_and_ghosts()
            self.last_room_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stage = 1
        self.time_left = self.max_steps
        self.last_room_idx = 0
        self.particles = []

        self._generate_stage()
        self._reset_player_and_ghosts()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # Return a terminal state
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1  # Base survival reward per step

        # --- 1. Handle Player Input ---
        movement = action[0]
        self.player["is_hiding"] = False
        
        # Check for hiding action
        can_hide = False
        for spot in self.hiding_spots:
            if self.player["rect"].colliderect(spot):
                can_hide = True
                break
        
        if movement == 2 and can_hide: # Down to hide
            self.player["is_hiding"] = True
            self.player["vx"] = 0
            reward -= 0.2 # Penalty for hiding
        else:
            # Horizontal movement
            if movement == 3: # Left
                self.player["vx"] = -self.PLAYER_SPEED
            elif movement == 4: # Right
                self.player["vx"] = self.PLAYER_SPEED
            else: # No horizontal input
                self.player["vx"] *= 0.8 # Friction
            
            # Jumping
            if movement == 1 and self.player["on_ground"]: # Up to jump
                self.player["vy"] = -self.JUMP_POWER
                self.player["on_ground"] = False
                # Sound: Jump
                for _ in range(5): # Dust particles on jump
                    self.particles.append(self._create_particle(self.player["rect"].midbottom, self.COLOR_PLATFORM))

        # --- 2. Update Physics & Player ---
        if not self.player["is_hiding"]:
            # Gravity
            self.player["vy"] += self.GRAVITY
            
            # Vertical movement and collision
            self.player["rect"].y += self.player["vy"]
            self.player["on_ground"] = False
            for plat in self.platforms:
                if self.player["rect"].colliderect(plat):
                    if self.player["vy"] > 0: # Falling down
                        self.player["rect"].bottom = plat.top
                        self.player["vy"] = 0
                        self.player["on_ground"] = True
                    elif self.player["vy"] < 0: # Hitting ceiling
                        self.player["rect"].top = plat.bottom
                        self.player["vy"] = 0
            
            # Horizontal movement and collision
            self.player["rect"].x += self.player["vx"]
            for plat in self.platforms:
                if self.player["rect"].colliderect(plat):
                    if self.player["vx"] > 0: # Moving right
                        self.player["rect"].right = plat.left
                    elif self.player["vx"] < 0: # Moving left
                        self.player["rect"].left = plat.right
            
            # World bounds
            self.player["rect"].left = max(0, self.player["rect"].left)
            self.player["rect"].right = min(self.WORLD_W, self.player["rect"].right)

        # --- 3. Update Ghosts ---
        is_player_running = abs(self.player["vx"]) > 1
        for ghost in self.ghosts:
            dist_to_player = math.hypot(ghost["rect"].centerx - self.player["rect"].centerx, ghost["rect"].centery - self.player["rect"].centery)
            
            is_alerted = dist_to_player < self.GHOST_ALERT_RADIUS and not self.player["is_hiding"] and is_player_running
            
            if is_alerted:
                ghost["state"] = "chase"
                ghost["target"] = self.player["rect"].center
            elif ghost["state"] == "chase" and (dist_to_player > self.GHOST_ALERT_RADIUS * 1.5 or self.player["is_hiding"]):
                ghost["state"] = "patrol"
                ghost["target"] = None

            if ghost["state"] == "patrol" and ghost["target"] is None:
                # Pick a new random point in home room
                home_x = ghost["home_room_idx"] * self.W
                target_x = home_x + self.np_random.uniform(50, self.W - 50)
                target_y = self.np_random.uniform(50, self.H - 50)
                ghost["target"] = (target_x, target_y)

            # Move towards target
            if ghost["target"]:
                dx = ghost["target"][0] - ghost["rect"].centerx
                dy = ghost["target"][1] - ghost["rect"].centery
                dist = math.hypot(dx, dy)
                if dist < ghost["speed"]:
                    if ghost["state"] == "patrol":
                        ghost["target"] = None # Reached patrol point
                else:
                    ghost["rect"].x += (dx / dist) * ghost["speed"]
                    ghost["rect"].y += (dy / dist) * ghost["speed"]

            # Ghost particle trail
            if self.steps % 3 == 0:
                self.particles.append(self._create_particle(ghost["rect"].center, self.COLOR_GHOST, lifespan=15, size=self.np_random.uniform(2, 5)))

        # --- 4. Update Particles ---
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["lifespan"] -= 1

        # --- 5. Check Game Events & Termination ---
        terminated = False
        
        # Player caught by ghost
        for ghost in self.ghosts:
            if self.player["rect"].colliderect(ghost["rect"]) and not self.player["is_hiding"]:
                self.game_over = True
                terminated = True
                reward = -100
                # Sound: Player caught
                break
        if terminated: # Early exit if caught
            self.score += reward
            return self._get_observation(), reward, terminated, False, self._get_info()
            
        # Player reached exit
        if self.player["rect"].colliderect(self.exit_door):
            stage_completion_reward = 5
            self.score += stage_completion_reward
            reward += stage_completion_reward
            self._advance_stage()
            # Sound: Stage complete
            if self.win:
                win_reward = 100
                self.score += win_reward
                reward += win_reward
                terminated = True
                self.game_over = True
        
        # Player entered new room
        current_room_idx = int(self.player["rect"].centerx // self.W)
        if current_room_idx > self.last_room_idx:
            room_reward = 1
            self.score += room_reward
            reward += room_reward
            self.last_room_idx = current_room_idx
            # Sound: New room entered
            # When player enters a new room, reset ghosts of previous rooms to patrol
            for g in self.ghosts:
                if g["home_room_idx"] < current_room_idx:
                    g["state"] = "patrol"
                    g["target"] = None

        # Timer runs out
        self.time_left -= 1
        if self.time_left <= 0:
            self.game_over = True
            terminated = True
            reward = -50  # Timeout penalty
            # Sound: Time up

        self.steps += 1
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particle(self, pos, color, lifespan=20, size=3):
        return {
            "x": pos[0], "y": pos[1],
            "vx": self.np_random.uniform(-1, 1), "vy": self.np_random.uniform(-1, 1),
            "lifespan": lifespan, "max_lifespan": lifespan,
            "color": color, "size": size
        }

    def _get_observation(self):
        # Camera follows player, clamped to world bounds
        self.camera_x = self.player["rect"].centerx - self.W / 2
        self.camera_x = max(0, min(self.WORLD_W - self.W, self.camera_x))

        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Flicker effect
        self.flicker_alpha = 20 + math.sin(self.steps * 0.2) * 15
        flicker_surface = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        flicker_surface.fill((0, 0, 0, self.flicker_alpha))
        self.screen.blit(flicker_surface, (0, 0))

        # Render all game elements relative to camera
        self._render_game()
        
        # Render UI overlay (not affected by camera)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw hiding spots
        for spot in self.hiding_spots:
            # Draw a darker rect as the main body
            draw_rect = spot.move(-self.camera_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_HIDING_SPOT, draw_rect)
            # Draw vertical lines for texture
            for i in range(int(draw_rect.left), int(draw_rect.right), 8):
                pygame.draw.line(self.screen, self.COLOR_HIDING_SPOT_LINE, (i, draw_rect.top), (i, draw_rect.bottom), 2)

        # Draw platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat.move(-self.camera_x, 0))

        # Draw exit door
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_door.move(-self.camera_x, 0))
        
        # Draw ghosts
        for ghost in self.ghosts:
            r = ghost["rect"].move(-self.camera_x, 0)
            bob = math.sin(ghost["bob_phase"] + self.steps * 0.1) * 3
            # Ghost shape
            points = [
                (r.left, r.bottom), (r.left, r.top + r.height/2),
                (r.centerx, r.top + bob), (r.right, r.top + r.height/2),
                (r.right, r.bottom), (r.right - r.width/4, r.bottom - r.height/4),
                (r.centerx, r.bottom), (r.left + r.width/4, r.bottom - r.height/4)
            ]
            points = [(int(px), int(py)) for px, py in points]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GHOST)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GHOST)
            # Eyes
            eye_y = r.top + r.height/2 - 5 + bob
            pygame.draw.circle(self.screen, self.COLOR_BG, (int(r.centerx - 5), int(eye_y)), 3)
            pygame.draw.circle(self.screen, self.COLOR_BG, (int(r.centerx + 5), int(eye_y)), 3)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = (*p["color"], alpha)
            size = int(p["size"])
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p["x"] - self.camera_x - size), int(p["y"] - size)))

        # Draw player
        player_color = self.COLOR_PLAYER
        player_rect = self.player["rect"].copy()
        if self.player["is_hiding"]:
            player_color = (self.COLOR_PLAYER[0]//2, self.COLOR_PLAYER[1]//2, self.COLOR_PLAYER[2]//2, 180)
            player_rect.height = self.PLAYER_H * 0.7
            player_rect.bottom = self.player["rect"].bottom
            temp_surf = pygame.Surface(player_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, player_color, (0, 0, *player_rect.size), border_radius=4)
            self.screen.blit(temp_surf, player_rect.move(-self.camera_x, 0).topleft)
        else:
            pygame.draw.rect(self.screen, player_color, player_rect.move(-self.camera_x, 0), border_radius=4)

    def _render_ui(self):
        # Stage display
        stage_text = self.font_ui.render(f"STAGE: {self.stage}/{self.NUM_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Timer display
        time_sec = self.time_left / self.FPS
        timer_color = self.COLOR_TEXT
        if time_sec < 10: timer_color = self.COLOR_TIMER_DANGER
        elif time_sec < 30: timer_color = self.COLOR_TIMER_WARN
        time_text = self.font_ui.render(f"TIME: {int(time_sec):02d}", True, timer_color)
        self.screen.blit(time_text, (self.W - time_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg = "YOU ESCAPED!"
                color = self.COLOR_EXIT
            elif self.time_left <= 0:
                msg = "TIME'S UP"
                color = self.COLOR_TIMER_DANGER
            else:
                msg = "CAUGHT!"
                color = self.COLOR_GHOST
            
            end_text = self.font_game_over.render(msg, True, color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.W / 2, self.H / 2)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_left_seconds": round(self.time_left / self.FPS, 2)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Reset is required before get_observation
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.H, self.W, 3), f"Obs shape is {obs.shape}, expected {(self.H, self.W, 3)}"
        assert obs.dtype == np.uint8
        
        # Test reset return types
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to run the game directly to test it
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Haunted House Escape")
    clock = pygame.time.Clock()
    
    running = True
    
    while running:
        # Game loop
        done = False
        total_reward = 0
        action = [0, 0, 0] 

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    running = False
            
            keys = pygame.key.get_pressed()
            
            # Reset action
            # movement: 0=none, 1=up, 2=down, 3=left, 4=right
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            else:
                action[0] = 0

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(env.FPS)

        # Post-game screen
        print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
        
        if running:
            # Wait for user to restart or quit
            wait_for_input = True
            while wait_for_input:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_input = False
                        running = False
                    if event.type == pygame.KEYDOWN:
                        wait_for_input = False # Any key press restarts
                clock.tick(15)
            
            if running:
                obs, info = env.reset()

    env.close()