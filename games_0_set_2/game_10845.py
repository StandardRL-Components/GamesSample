import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:04:57.197695
# Source Brief: brief_00845.md
# Brief Index: 845
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Rhythmic Gymnastics Environment

    The agent controls a character on a performance platform.
    The goal is to leap through a sequence of rhythmically appearing portals.
    Success requires precise timing and positioning, rewarding fluid and accurate movement.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Leap through a sequence of rhythmically appearing portals. "
        "Success requires precise timing and positioning to match the beat."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move on the platform. "
        "Press space to jump through the portals on the beat."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLATFORM_RECT = pygame.Rect(80, 280, 480, 100)
    FPS = 30

    # Colors
    COLOR_BG_TOP = (10, 5, 30)
    COLOR_BG_BOTTOM = (40, 10, 60)
    COLOR_PLATFORM = (60, 30, 90)
    COLOR_PLATFORM_EDGE = (120, 90, 180)
    COLOR_PLAYER = (220, 255, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255)
    COLOR_TEXT = (255, 255, 255)
    PORTAL_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 80, 255),   # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
    ]

    # Physics & Gameplay
    PLAYER_SPEED = 6.0
    PLAYER_DAMPING = 0.85
    PLAYER_JUMP_VEL = 18.0
    PLAYER_GRAVITY = 0.8
    PLAYER_SIZE = 12
    PORTAL_RADIUS = 25
    JUMP_SUCCESS_RADIUS = 35
    BEAT_WINDOW = 0.2  # 20% of beat duration is considered "on beat"
    MAX_STEPS = 1500

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self._pre_render_background()

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_reward_info = ""

        # Player state
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_z = 0  # Height off the platform
        self.player_z_vel = 0
        self.is_jumping = False

        # Rhythm and routine state
        self.beat_duration_steps = int(self.FPS * 1.0) # 1 second per beat initially
        self.beat_timer = 0
        self.current_beat_idx = 0
        self.routine = []
        self.routine_progress = 0
        self.successful_routines = 0
        self.active_portal_idx = -1
        self.jump_initiated_beat = -1

        # Input state
        self.prev_space_held = False

        # Effects
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player_pos = pygame.math.Vector2(self.PLATFORM_RECT.centerx, self.PLATFORM_RECT.centery)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_z = 0
        self.player_z_vel = 0
        self.is_jumping = False

        # Rhythm and routine state
        difficulty_factor = 1.0 - (self.successful_routines // 5) * 0.05
        self.beat_duration_steps = max(int(self.FPS * 0.4), int(self.FPS * difficulty_factor))
        self.beat_timer = 0
        self.current_beat_idx = 0
        self.routine = self._generate_routine()
        self.routine_progress = 0
        self.active_portal_idx = -1
        self.jump_initiated_beat = -1

        # Input state
        self.prev_space_held = False

        # Effects
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_player_input(movement, space_held)

        # --- Game Logic Update ---
        self._update_player_state()
        self._update_rhythm_and_portals()
        self._update_particles()

        # --- Reward and Termination ---
        reward = self._calculate_reward()
        self.score += reward
        self.steps += 1
        terminated = self._check_termination()
        self.game_over = terminated
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_routine(self):
        routine = []
        num_portals = 5 + self.successful_routines // 2
        beat_gap = 4
        
        # Ensure portals are reachable within the time
        max_dist_per_beat = self.PLAYER_SPEED * self.beat_duration_steps
        max_dist = max_dist_per_beat * beat_gap * 0.8 # 80% margin

        last_pos = self.player_pos.copy()
        for i in range(num_portals):
            while True:
                px = self.PLATFORM_RECT.left + self.PORTAL_RADIUS + self.np_random.random() * (self.PLATFORM_RECT.width - 2 * self.PORTAL_RADIUS)
                py = self.PLATFORM_RECT.top + self.PORTAL_RADIUS + self.np_random.random() * (self.PLATFORM_RECT.height - 2 * self.PORTAL_RADIUS)
                pos = pygame.math.Vector2(px, py)
                if pos.distance_to(last_pos) < max_dist:
                    break
            
            routine.append({
                "pos": pos,
                "beat": (i + 1) * beat_gap,
                "color": self.PORTAL_COLORS[i % len(self.PORTAL_COLORS)],
                "hit": False,
                "missed": False,
            })
            last_pos = pos
        return routine

    def _handle_player_input(self, movement, space_held):
        # --- Movement ---
        if not self.is_jumping:
            accel = pygame.math.Vector2(0, 0)
            if movement == 1: accel.y = -1  # Up
            elif movement == 2: accel.y = 1   # Down
            elif movement == 3: accel.x = -1  # Left
            elif movement == 4: accel.x = 1   # Right
            
            if accel.length() > 0:
                accel.scale_to_length(self.PLAYER_SPEED)
            self.player_vel = accel

        # --- Jump ---
        jump_triggered = space_held and not self.prev_space_held
        if jump_triggered and not self.is_jumping:
            # SFX: Jump_Sound()
            self.is_jumping = True
            self.player_z_vel = self.PLAYER_JUMP_VEL
            self.jump_initiated_beat = self.current_beat_idx
            self.player_vel *= 0.5 # Dampen horizontal movement on jump
            self._create_particles(self.player_pos, self.COLOR_PLAYER, 20, 5, 20)

        self.prev_space_held = space_held

    def _update_player_state(self):
        # Apply velocity and damping
        if not self.is_jumping:
             self.player_pos += self.player_vel
             self.player_vel *= self.PLAYER_DAMPING

        # Keep player on platform
        self.player_pos.x = max(self.PLATFORM_RECT.left, min(self.PLATFORM_RECT.right, self.player_pos.x))
        self.player_pos.y = max(self.PLATFORM_RECT.top, min(self.PLATFORM_RECT.bottom, self.player_pos.y))

        # Handle jump physics
        if self.is_jumping:
            self.player_z += self.player_z_vel
            self.player_z_vel -= self.PLAYER_GRAVITY
            if self.player_z < 0:
                # SFX: Land_Sound()
                self.player_z = 0
                self.is_jumping = False
                self.player_z_vel = 0
                self._create_particles(self.player_pos, self.COLOR_PLAYER_GLOW, 30, 3, 15, is_ground_burst=True)

    def _update_rhythm_and_portals(self):
        self.beat_timer += 1
        if self.beat_timer >= self.beat_duration_steps:
            self.beat_timer = 0
            self.current_beat_idx += 1

        self.active_portal_idx = -1
        for i, portal in enumerate(self.routine):
            if not portal["hit"] and not portal["missed"]:
                if portal["beat"] == self.current_beat_idx:
                    self.active_portal_idx = i
                # Mark portals as missed if we've passed their beat without a hit
                if self.current_beat_idx > portal["beat"] + 1:
                    portal["missed"] = True
                    # SFX: Miss_Sound()
                break # Only one portal can be active

    def _calculate_reward(self):
        reward = 0
        
        # --- Event-based rewards (checked on landing) ---
        if self.is_jumping is False and self.jump_initiated_beat != -1:
            jump_beat = self.jump_initiated_beat
            self.jump_initiated_beat = -1 # Consume the jump event
            
            target_portal = None
            for p in self.routine:
                if p["beat"] == jump_beat:
                    target_portal = p
                    break
            
            if target_portal:
                dist = self.player_pos.distance_to(target_portal["pos"])
                if dist < self.JUMP_SUCCESS_RADIUS:
                    # SFX: Success_Chime()
                    reward += 10.0
                    target_portal["hit"] = True
                    self.routine_progress += 1
                    self.last_reward_info = "Portal Hit! +10"
                    self._create_particles(target_portal["pos"], target_portal["color"], 50, 8, 30)
                else:
                    reward -= 5.0
                    target_portal["missed"] = True
                    self.last_reward_info = "Jump Missed! -5"
            else: # Jumped when no portal was active
                reward -= 1.0
                self.last_reward_info = "Bad Timing! -1"

        # --- Continuous rewards ---
        if self.active_portal_idx != -1:
            portal = self.routine[self.active_portal_idx]
            dist_to_portal = self.player_pos.distance_to(portal["pos"])
            
            # Reward for being near the active portal
            if dist_to_portal < self.PORTAL_RADIUS * 2:
                reward += 1.0 * (1 - dist_to_portal / (self.PORTAL_RADIUS * 2))

            # Penalty for moving away from active portal
            if self.player_vel.length() > 0.1:
                dir_to_portal = (portal["pos"] - self.player_pos).normalize()
                if self.player_vel.dot(dir_to_portal) < 0:
                    reward -= 0.1

        return reward

    def _check_termination(self):
        # Win condition: all portals hit
        if self.routine_progress == len(self.routine) and len(self.routine) > 0:
            self.score += 100 # Bonus for completing routine
            self.successful_routines += 1
            self.last_reward_info = f"Routine Complete! +100"
            return True

        # Lose condition: any portal definitively missed
        if any(p["missed"] for p in self.routine):
             return True

        return False

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "successful_routines": self.successful_routines,
            "routine_progress": f"{self.routine_progress}/{len(self.routine) if self.routine else 0}",
        }

    def _pre_render_background(self):
        self.bg_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.bg_surface, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # --- Render Platform ---
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, self.PLATFORM_RECT, border_radius=10)
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM_EDGE, self.PLATFORM_RECT, width=2, border_radius=10)

        # --- Render Portals ---
        self._render_portals()

        # --- Render Particles ---
        self._render_particles()

        # --- Render Player ---
        self._render_player()

    def _render_player(self):
        # Shadow
        shadow_size = int(self.PLAYER_SIZE * (1 - self.player_z / 200))
        shadow_alpha = 90 - self.player_z / 2
        if shadow_size > 0 and shadow_alpha > 0:
            shadow_surf = pygame.Surface((shadow_size * 2, shadow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(shadow_surf, (0, 0, 0, shadow_alpha), (shadow_size, shadow_size), shadow_size)
            self.screen.blit(shadow_surf, self.player_pos - pygame.math.Vector2(shadow_size, shadow_size))

        # Player position adjusted for jump height
        render_pos = self.player_pos - pygame.math.Vector2(0, self.player_z)
        size = int(self.PLAYER_SIZE + self.player_z / 15)

        # Glow effect
        glow_size = int(size * 2.5)
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 30), (glow_size, glow_size), glow_size)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 60), (glow_size, glow_size), int(glow_size * 0.7))
        self.screen.blit(glow_surf, render_pos - pygame.math.Vector2(glow_size, glow_size))

        # Core player
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (int(render_pos.x), int(render_pos.y)), size)

        # Beat indicator
        beat_progress = self.beat_timer / self.beat_duration_steps
        indicator_radius = int(size * 1.5 + 40 * (1 - beat_progress))
        indicator_alpha = 150 * beat_progress
        if self.active_portal_idx != -1:
             pygame.gfxdraw.aacircle(self.screen, int(render_pos.x), int(render_pos.y), indicator_radius, (*self.COLOR_PLAYER_GLOW, indicator_alpha))

    def _render_portals(self):
        for i, portal in enumerate(self.routine):
            if portal["hit"] or portal["missed"]:
                continue

            # Pulse effect for upcoming portals
            pulse = 0
            alpha = 50
            width = 2
            
            # Active portal is bright and large
            if i == self.active_portal_idx:
                beat_progress = self.beat_timer / self.beat_duration_steps
                pulse = abs(math.sin(beat_progress * math.pi)) * 8
                alpha = 255
                width = 5
            # Next portal in sequence is highlighted
            elif i > 0 and self.routine[i-1]["hit"] and not portal["hit"]:
                 alpha = 150
                 width = 3
                 pulse = abs(math.sin(self.steps * 0.1)) * 3


            color = (*portal["color"], alpha)
            radius = int(self.PORTAL_RADIUS + pulse)
            
            # Draw anti-aliased circle
            pygame.gfxdraw.aacircle(self.screen, int(portal["pos"].x), int(portal["pos"].y), radius, color)
            pygame.gfxdraw.filled_circle(self.screen, int(portal["pos"].x), int(portal["pos"].y), radius - width, (*color[:3], int(alpha/10)))


    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['lifespan'] / p['max_lifespan']))
            if size > 0:
                # Simple square particles are fast and look good for this style
                rect = pygame.Rect(p['pos'].x - size//2, p['pos'].y - size//2, size, size)
                surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                surf.fill(color)
                self.screen.blit(surf, rect.topleft)


    def _create_particles(self, pos, color, count, max_speed, max_lifespan, is_ground_burst=False):
        for _ in range(count):
            if is_ground_burst:
                angle = self.np_random.random() * math.pi # Only upward hemisphere
                speed = 1 + self.np_random.random() * max_speed
                vel = pygame.math.Vector2(math.cos(angle) * speed, -math.sin(angle) * speed)
            else:
                angle = self.np_random.random() * 2 * math.pi
                speed = 1 + self.np_random.random() * max_speed
                vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': int(max_lifespan * 0.5 + self.np_random.random() * max_lifespan * 0.5),
                'max_lifespan': max_lifespan,
                'color': color,
                'size': 2 + self.np_random.random() * 4
            })


    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Routine progress
        progress_text = self.font_main.render(f"Routine: {self.routine_progress}/{len(self.routine) if self.routine else 0}", True, self.COLOR_TEXT)
        self.screen.blit(progress_text, (self.SCREEN_WIDTH - progress_text.get_width() - 10, 10))
        
        # Game Over Message
        if self.game_over:
            is_win = self.routine_progress == len(self.routine) and len(self.routine) > 0
            msg = "ROUTINE COMPLETE" if is_win else "SEQUENCE FAILED"
            color = (150, 255, 150) if is_win else (255, 150, 150)
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,150))
            self.screen.blit(overlay, (0,0))
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Example ---
    # Set a real video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a persistent display window
    pygame.display.set_caption("Rhythmic Gymnastics Gym Environment")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    done = False
    total_reward = 0
    
    while not done:
        # --- Action Mapping for Human ---
        action = [0, 0, 0] # Default no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to get it to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(GameEnv.FPS)

        if done:
            print(f"Episode finished. Total Reward: {total_reward}")
            print(f"Final Info: {info}")
            # Wait a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False
            total_reward = 0

    env.close()