import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:22:55.464075
# Source Brief: brief_00375.md
# Brief Index: 375
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Infiltrate a cyberpunk facility as a stealth agent. Use holographic clones to distract guards "
        "and eliminate all targets to reach the extraction point."
    )
    user_guide = (
        "Controls: ←→ to move, ↑ to jump, ↓ to crouch. Press space to deploy a clone and shift to use a smoke bomb."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000

        # Colors (Cyberpunk Palette)
        self.COLOR_BG = (10, 2, 25)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_CLONE = (0, 150, 255)
        self.COLOR_CLONE_GLOW = (0, 150, 255, 40)
        self.COLOR_GUARD = (255, 50, 50)
        self.COLOR_GUARD_GLOW = (255, 50, 50, 60)
        self.COLOR_GUARD_VISION = (255, 255, 0, 40)
        self.COLOR_TARGET = (255, 255, 0)
        self.COLOR_EXTRACTION = (150, 0, 255)
        self.COLOR_PLATFORM = (50, 50, 80)
        self.COLOR_PLATFORM_LINE = (100, 100, 150)
        self.COLOR_SMOKE = (100, 100, 100, 10)
        self.COLOR_EMP = (0, 150, 255, 150)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_UI_BG = (30, 30, 60, 180)

        # Physics & Gameplay
        self.GRAVITY = 0.8
        self.PLAYER_SPEED = 4.0
        self.JUMP_STRENGTH = -12.0
        self.CLONE_COOLDOWN_STEPS = 90  # 3 seconds at 30 FPS
        self.CLONE_BASE_DURATION = 150 # 5 seconds
        self.GADGET_COOLDOWN_STEPS = 150 # 5 seconds

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward = 0
        
        self.player = None
        self.platforms = []
        self.guards = []
        self.targets = []
        self.clones = []
        self.particles = []
        self.extraction_point = None
        self.camera_offset = pygame.math.Vector2(0, 0)

        self.clone_cooldown = 0
        self.clone_duration_bonus = 0
        self.gadget_cooldown = 0
        self.gadget_charges = 1 # Start with one smoke bomb
        self.emp_unlocked = False

        self.guard_speed_bonus = 0.0

        self.last_action = np.array([0, 0, 0])
        self.world_width = 0

    def _generate_level(self):
        self.platforms = []
        self.targets = []
        self.guards = []
        
        # Ground floor
        self.world_width = self.WIDTH * 3
        self.platforms.append(pygame.Rect(0, self.HEIGHT - 20, self.world_width, 20))
        
        # Procedural platforms
        current_x = 150
        while current_x < self.world_width - 300:
            width = self.np_random.integers(100, 250)
            height = self.np_random.integers(self.HEIGHT - 200, self.HEIGHT - 50)
            gap = self.np_random.integers(80, 150)
            
            platform_rect = pygame.Rect(current_x, height, width, 15)
            self.platforms.append(platform_rect)
            
            # Place a target or guard on the platform
            if self.np_random.random() > 0.6:
                target_pos = pygame.math.Vector2(platform_rect.centerx, platform_rect.top - 15)
                self.targets.append({"pos": target_pos, "active": True})
            elif self.np_random.random() > 0.5:
                guard_start = platform_rect.left + 20
                guard_end = platform_rect.right - 20
                guard_pos = pygame.math.Vector2(guard_start, platform_rect.top - 20)
                self.guards.append(self._create_guard(guard_pos, guard_start, guard_end))

            current_x += width + gap
            
        # Ensure at least one target
        if not self.targets:
            p = self.platforms[-1]
            self.targets.append({"pos": pygame.math.Vector2(p.centerx, p.top - 15), "active": True})

        # Extraction point
        last_platform = self.platforms[-1]
        self.extraction_point = pygame.Rect(last_platform.right - 60, last_platform.top - 60, 40, 40)

    def _create_guard(self, pos, patrol_start, patrol_end):
        return {
            "pos": pos,
            "size": 20,
            "speed": 1.0 + self.guard_speed_bonus,
            "patrol_start": patrol_start,
            "patrol_end": patrol_end,
            "direction": 1,
            "state": "patrol", # patrol, investigate, alerted
            "vision_range": 150,
            "vision_angle": 45,
            "alert_timer": 0,
            "investigate_target": None,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward = 0
        
        self.player = {
            "pos": pygame.math.Vector2(100, self.HEIGHT - 100),
            "vel": pygame.math.Vector2(0, 0),
            "size": pygame.math.Vector2(20, 40),
            "on_ground": False,
            "is_crouching": False,
        }
        
        self.clones.clear()
        self.particles.clear()
        
        self.clone_cooldown = 0
        self.gadget_cooldown = 0
        
        # Progression reset
        self.clone_duration_bonus = 0
        self.gadget_charges = 1
        self.emp_unlocked = False
        self.guard_speed_bonus = 0.0

        self._generate_level()

        self.last_action = np.array([0, 0, 0])
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        self.reward = 0
        
        self._handle_player_input(movement, space_pressed, shift_pressed)
        self._update_player()
        self._update_guards()
        self._update_clones()
        self._update_particles()
        self._update_camera()
        
        self._check_interactions()
        self._update_progression()

        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True


        # Continuous reward for survival
        if not terminated:
            self.reward += 0.01 

        self.last_action = action
        
        return (
            self._get_observation(),
            self.reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_pressed, shift_pressed):
        # Crouching
        self.player["is_crouching"] = (movement == 0 or movement == 2)

        # Horizontal Movement
        if movement == 3: # Left
            self.player["vel"].x = -self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player["vel"].x = self.PLAYER_SPEED
        else:
            self.player["vel"].x = 0

        # Jumping
        if movement == 1 and self.player["on_ground"]:
            self.player["vel"].y = self.JUMP_STRENGTH
            self.player["on_ground"] = False
            # sfx: player_jump.wav

        # Clone Deployment (on press)
        clone_press = space_pressed and not (self.last_action[1] == 1)
        if clone_press and self.clone_cooldown <= 0:
            self.clone_cooldown = self.CLONE_COOLDOWN_STEPS
            clone_pos = pygame.math.Vector2(self.player["pos"])
            clone_duration = self.CLONE_BASE_DURATION + self.clone_duration_bonus
            self.clones.append({"pos": clone_pos, "timer": clone_duration})
            # sfx: clone_deploy.wav
            # Create a noise event for guards
            for guard in self.guards:
                if guard["pos"].distance_to(clone_pos) < 250 and guard["state"] == "patrol":
                    guard["state"] = "investigate"
                    guard["investigate_target"] = clone_pos
                    self.reward += 1 # Reward for distraction

        # Gadget Use (on press)
        gadget_press = shift_pressed and not (self.last_action[2] == 1)
        if gadget_press and self.gadget_cooldown <= 0 and self.gadget_charges > 0:
            self.gadget_cooldown = self.GADGET_COOLDOWN_STEPS
            self.gadget_charges -= 1
            # sfx: smoke_bomb.wav
            # Create smoke particles
            for _ in range(100):
                offset = pygame.math.Vector2(self.np_random.uniform(-50, 50), self.np_random.uniform(-50, 50))
                self.particles.append({
                    "pos": self.player["pos"] + offset,
                    "type": "smoke",
                    "timer": self.np_random.integers(60, 120),
                    "radius": self.np_random.uniform(10, 30)
                })

    def _update_player(self):
        # Apply gravity
        self.player["vel"].y += self.GRAVITY
        if self.player["vel"].y > 10: self.player["vel"].y = 10

        # Move and handle collisions
        self.player["pos"].x += self.player["vel"].x
        self.player["pos"].y += self.player["vel"].y

        # Update player rect based on crouching
        player_height = self.player["size"].y / 2 if self.player["is_crouching"] else self.player["size"].y
        player_rect = pygame.Rect(self.player["pos"].x - self.player["size"].x / 2, 
                                  self.player["pos"].y - player_height, 
                                  self.player["size"].x, player_height)

        self.player["on_ground"] = False
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                # Vertical collision
                if self.player["vel"].y > 0 and player_rect.bottom > plat.top and player_rect.bottom < plat.top + self.player["vel"].y + 1:
                    player_rect.bottom = plat.top
                    self.player["pos"].y = player_rect.bottom
                    self.player["vel"].y = 0
                    self.player["on_ground"] = True
                # Horizontal collision (simple)
                elif self.player["vel"].x > 0 and player_rect.right > plat.left:
                     player_rect.right = plat.left
                     self.player["pos"].x = player_rect.centerx
                elif self.player["vel"].x < 0 and player_rect.left < plat.right:
                     player_rect.left = plat.right
                     self.player["pos"].x = player_rect.centerx

        # World boundaries
        self.player["pos"].x = max(self.player["size"].x/2, min(self.player["pos"].x, self.world_width - self.player["size"].x/2))
        if self.player["pos"].y > self.HEIGHT + 100: # Fell off world
            self.game_over = True
            self.reward = -100

    def _update_guards(self):
        player_rect = self._get_player_rect()

        for guard in self.guards:
            # State logic
            if guard["state"] == "patrol":
                if guard["pos"].x <= guard["patrol_start"]: guard["direction"] = 1
                if guard["pos"].x >= guard["patrol_end"]: guard["direction"] = -1
                guard["pos"].x += guard["speed"] * guard["direction"]
            
            elif guard["state"] == "investigate":
                if guard["investigate_target"] and guard["pos"].distance_to(guard["investigate_target"]) < 20:
                    guard["state"] = "patrol"
                elif guard["investigate_target"]:
                    dir_vec = (guard["investigate_target"] - guard["pos"]).normalize()
                    guard["pos"] += dir_vec * guard["speed"] * 1.5 # Move faster
                else: # Failsafe
                    guard["state"] = "patrol"

            elif guard["state"] == "alerted":
                guard["alert_timer"] -= 1
                if guard["alert_timer"] <= 0:
                    guard["state"] = "patrol"
            
            # Vision check
            if guard["state"] != "alerted":
                detected = self._is_player_in_vision(guard, player_rect)
                if detected:
                    guard["state"] = "alerted"
                    guard["alert_timer"] = 180 # 6 seconds
                    # sfx: guard_alert.wav

            # Player collision
            guard_rect = pygame.Rect(guard["pos"].x - guard["size"]/2, guard["pos"].y - guard["size"], guard["size"], guard["size"])
            if player_rect.colliderect(guard_rect) and guard["state"] == "alerted":
                self.game_over = True
                self.reward = -100
                # sfx: player_death.wav

    def _is_player_in_vision(self, guard, player_rect):
        # Check if any smoke is blocking vision
        for p in self.particles:
            if p["type"] == "smoke":
                if pygame.Rect(p["pos"].x-p["radius"], p["pos"].y-p["radius"], p["radius"]*2, p["radius"]*2).clipline(guard["pos"], player_rect.center):
                    return False

        dist_to_player = guard["pos"].distance_to(player_rect.center)
        if dist_to_player > guard["vision_range"]:
            return False

        if dist_to_player > 0:
            dir_to_player = (pygame.math.Vector2(player_rect.center) - guard["pos"]).normalize()
        else: # Player is on top of guard
            return True

        guard_facing_dir = pygame.math.Vector2(guard["direction"], 0)
        
        angle = guard_facing_dir.angle_to(dir_to_player)
        if abs(angle) > guard["vision_angle"]:
            return False

        # Raycast to check for platform occlusion
        for plat in self.platforms:
            if plat.clipline(guard["pos"], player_rect.center):
                return False
        
        return True

    def _update_clones(self):
        self.clones[:] = [c for c in self.clones if c["timer"] > 0]
        for clone in self.clones:
            clone["timer"] -= 1

    def _update_particles(self):
        self.particles[:] = [p for p in self.particles if p["timer"] > 0]
        for p in self.particles:
            p["timer"] -= 1

    def _update_camera(self):
        self.camera_offset.x = self.player["pos"].x - self.WIDTH / 2
        self.camera_offset.x = max(0, min(self.camera_offset.x, self.world_width - self.WIDTH))
        self.camera_offset.y = 0

    def _check_interactions(self):
        player_rect = self._get_player_rect()
        
        # Target elimination
        for target in self.targets:
            if target["active"] and player_rect.collidepoint(target["pos"]):
                target["active"] = False
                self.score += 25
                self.reward += 5
                # sfx: target_eliminated.wav

        # Extraction
        active_targets = any(t["active"] for t in self.targets)
        if not active_targets and self.extraction_point and player_rect.colliderect(self.extraction_point):
            self.score += 100
            self.reward = 100
            self.game_over = True
            # sfx: mission_complete.wav

    def _update_progression(self):
        if not self.emp_unlocked and self.score >= 500:
            self.emp_unlocked = True # Not used, but for future expansion
        
        new_clone_bonus = int(self.score / 200) * 30 # 1 sec per 200 pts
        if new_clone_bonus > self.clone_duration_bonus:
            self.clone_duration_bonus = new_clone_bonus

        new_guard_speed_bonus = (self.score // 500) * 0.05
        if new_guard_speed_bonus > self.guard_speed_bonus:
            self.guard_speed_bonus = new_guard_speed_bonus
            for guard in self.guards: # Update existing guards
                guard["speed"] = 1.0 + self.guard_speed_bonus

    def _check_termination(self):
        return self.game_over

    def _get_player_rect(self):
        player_height = self.player["size"].y / 2 if self.player["is_crouching"] else self.player["size"].y
        return pygame.Rect(self.player["pos"].x - self.player["size"].x / 2, 
                           self.player["pos"].y - player_height, 
                           self.player["size"].x, player_height)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw platforms
        for plat in self.platforms:
            p = plat.move(-self.camera_offset.x, -self.camera_offset.y)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_LINE, p, 1)

        # Draw extraction point
        if self.extraction_point:
            active_targets = any(t["active"] for t in self.targets)
            ext_color = self.COLOR_EXTRACTION if not active_targets else self.COLOR_UI_BG
            pygame.draw.rect(self.screen, ext_color, self.extraction_point.move(-self.camera_offset))

        # Draw targets
        for target in self.targets:
            if target["active"]:
                pos = target["pos"] - self.camera_offset
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 10, self.COLOR_TARGET)
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 10, self.COLOR_TARGET)

        # Draw particles (smoke)
        for p in self.particles:
            if p["type"] == "smoke":
                pos = p["pos"] - self.camera_offset
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(p["radius"]), self.COLOR_SMOKE)
        
        # Draw clones
        for clone in self.clones:
            pos = clone["pos"] - self.camera_offset
            size = self.player["size"]
            rect = pygame.Rect(pos.x - size.x/2, pos.y - size.y, size.x, size.y)
            pygame.draw.rect(self.screen, self.COLOR_CLONE, rect, border_radius=3)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y - size.y/2), int(size.x), self.COLOR_CLONE_GLOW)

        # Draw guards and their vision
        for guard in self.guards:
            pos = guard["pos"] - self.camera_offset
            size = guard["size"]
            # Vision cone
            if guard["state"] != "alerted":
                p1 = pos + pygame.math.Vector2(0, -size)
                p2 = p1 + pygame.math.Vector2(guard["direction"], 0).rotate(-guard["vision_angle"]) * guard["vision_range"]
                p3 = p1 + pygame.math.Vector2(guard["direction"], 0).rotate(guard["vision_angle"]) * guard["vision_range"]
                pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_GUARD_VISION)
                pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_GUARD_VISION)
            # Guard body
            rect = pygame.Rect(pos.x - size/2, pos.y - size, size, size)
            pygame.draw.rect(self.screen, self.COLOR_GUARD, rect, border_radius=3)
            # Glow/Alert indicator
            glow_color = self.COLOR_GUARD_GLOW if guard["state"] != "alerted" else (255, 0, 0, 150)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y - size/2), int(size*1.5), glow_color)


        # Draw player
        player_rect_cam = self._get_player_rect().move(-self.camera_offset)
        pygame.gfxdraw.filled_circle(self.screen, int(player_rect_cam.centerx), int(player_rect_cam.centery), int(self.player["size"].x * 1.8), self.COLOR_PLAYER_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect_cam, border_radius=3)

    def _render_ui(self):
        ui_surface = pygame.Surface((self.WIDTH, 80), pygame.SRCALPHA)
        pygame.draw.rect(ui_surface, self.COLOR_UI_BG, (0, 0, self.WIDTH, 80))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        ui_surface.blit(score_text, (10, 10))

        # Targets
        active_targets = sum(1 for t in self.targets if t["active"])
        targets_text = self.font_ui.render(f"TARGETS: {active_targets}/{len(self.targets)}", True, self.COLOR_TEXT)
        ui_surface.blit(targets_text, (10, 35))

        # Clone Cooldown
        clone_status = "READY" if self.clone_cooldown <= 0 else f"{(self.clone_cooldown / self.FPS):.1f}s"
        clone_text = self.font_ui.render(f"CLONE [SPACE]: {clone_status}", True, self.COLOR_TEXT)
        ui_surface.blit(clone_text, (self.WIDTH - 200, 10))

        # Gadget Cooldown
        gadget_status = f"READY ({self.gadget_charges})" if self.gadget_cooldown <= 0 else f"{(self.gadget_cooldown / self.FPS):.1f}s"
        gadget_text = self.font_ui.render(f"SMOKE [SHIFT]: {gadget_status}", True, self.COLOR_TEXT)
        ui_surface.blit(gadget_text, (self.WIDTH - 200, 35))

        self.screen.blit(ui_surface, (0, 0))

        if self.game_over:
            won = not any(t["active"] for t in self.targets)
            msg = "MISSION COMPLETE" if won else "AGENT COMPROMISED"
            color = self.COLOR_PLAYER if won else self.COLOR_GUARD
            
            end_text = self.font_big.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "targets_left": sum(1 for t in self.targets if t["active"]),
            "player_pos": tuple(self.player["pos"]),
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Example usage:
    env = GameEnv(render_mode="rgb_array")
    
    # Manual play loop
    obs, info = env.reset()
    done = False
    
    # Use a different screen for rendering if playing manually
    # This requires removing the SDL_VIDEODRIVER="dummy" line
    manual_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Cyberpunk Stealth Platformer")
    
    total_reward = 0
    
    # Key mapping for manual control
    key_map = {
        pygame.K_w: 1, pygame.K_UP: 1,
        pygame.K_s: 2, pygame.K_DOWN: 2,
        pygame.K_a: 3, pygame.K_LEFT: 3,
        pygame.K_d: 4, pygame.K_RIGHT: 4,
    }

    while not done:
        movement_action = 0 # No-op/crouch
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break # Prioritize first found key
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = np.array([movement_action, space_action, shift_action])
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        manual_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    # Keep the final screen visible for a moment
    pygame.time.wait(3000)
    
    env.close()