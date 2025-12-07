import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:01:19.662528
# Source Brief: brief_03241.md
# Brief Index: 3241
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control two pixels to avoid an early collision. Guide them to a successful merge after the timer runs out."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to reverse the vertical or horizontal direction of both pixels."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- CRITICAL: Environment Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_STEPS = 1000  # ~33.3 seconds
        self.MERGE_TIME_SECONDS = 30
        self.MERGE_TIME_STEPS = self.MERGE_TIME_SECONDS * self.FPS

        # --- CRITICAL: Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Visual Design ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_P1 = (0, 255, 150)
        self.COLOR_P2 = (0, 150, 255)
        self.COLOR_MERGED = (0, 255, 255)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_COLLISION = (255, 50, 50)
        self.ACTION_FEEDBACK_V_COLOR = (200, 200, 50)
        self.ACTION_FEEDBACK_H_COLOR = (50, 200, 200)

        # --- Game Mechanics ---
        self.PIXEL_SIZE = 10
        self.SPEED1 = 150  # Pixels per second
        self.SPEED2 = 225  # Pixels per second
        
        # Calculate diagonal velocity components to maintain constant speed
        self.v_comp1 = self.SPEED1 / math.sqrt(2)
        self.v_comp2 = self.SPEED2 / math.sqrt(2)

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 28)
            self.font_msg = pygame.font.SysFont(None, 52)
            
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.merged = False
        self.p1_pos = np.zeros(2, dtype=np.float32)
        self.p1_vel = np.zeros(2, dtype=np.float32)
        self.p2_pos = np.zeros(2, dtype=np.float32)
        self.p2_vel = np.zeros(2, dtype=np.float32)
        self.merged_pos = np.zeros(2, dtype=np.float32)
        self.merged_anim_timer = 0
        self.collision_flash_timer = 0
        self.particles = []
        self.action_effects = []

        # self.reset() is called by the wrapper
        # self.validate_implementation() is for debugging, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.merged = False
        self.collision_flash_timer = 0
        self.particles.clear()
        self.action_effects.clear()

        # Place pixels far apart to avoid instant collision
        min_dist = 200
        while True:
            self.p1_pos = self.np_random.uniform(
                low=self.PIXEL_SIZE, 
                high=[self.SCREEN_WIDTH - self.PIXEL_SIZE, self.SCREEN_HEIGHT - self.PIXEL_SIZE], 
                size=2
            ).astype(np.float32)
            self.p2_pos = self.np_random.uniform(
                low=self.PIXEL_SIZE, 
                high=[self.SCREEN_WIDTH - self.PIXEL_SIZE, self.SCREEN_HEIGHT - self.PIXEL_SIZE], 
                size=2
            ).astype(np.float32)
            if np.linalg.norm(self.p1_pos - self.p2_pos) > min_dist:
                break
        
        # Set initial diagonal velocities
        self.p1_vel = np.array([
            self.np_random.choice([-self.v_comp1, self.v_comp1]), 
            self.np_random.choice([-self.v_comp1, self.v_comp1])
        ], dtype=np.float32)
        self.p2_vel = np.array([
            self.np_random.choice([-self.v_comp2, self.v_comp2]), 
            self.np_random.choice([-self.v_comp2, self.v_comp2])
        ], dtype=np.float32)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            # This allows the final screen to be displayed
            terminated = True
            truncated = self.steps >= self.MAX_STEPS
            return self._get_observation(), 0, terminated, truncated, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        
        if movement == 1 or movement == 2:  # Reverse vertical direction (mapped to UP/DOWN)
            self.p1_vel[1] *= -1
            self.p2_vel[1] *= -1
            self._add_action_effect(self.ACTION_FEEDBACK_V_COLOR)
        elif movement == 3 or movement == 4:  # Reverse horizontal direction (mapped to LEFT/RIGHT)
            self.p1_vel[0] *= -1
            self.p2_vel[0] *= -1
            self._add_action_effect(self.ACTION_FEEDBACK_H_COLOR)

        # --- Game Logic Update ---
        self.steps += 1
        reward = 0
        terminated = False
        truncated = False

        if not self.merged:
            # Update positions with wrap-around
            self.p1_pos = (self.p1_pos + self.p1_vel / self.FPS) % [self.SCREEN_WIDTH, self.SCREEN_HEIGHT]
            self.p2_pos = (self.p2_pos + self.p2_vel / self.FPS) % [self.SCREEN_WIDTH, self.SCREEN_HEIGHT]

            # Add trail particles
            self._add_particle(self.p1_pos, self.COLOR_P1)
            self._add_particle(self.p2_pos, self.COLOR_P2)

        # Update visual effects
        self._update_particles()
        self._update_action_effects()

        # --- Collision and Merge Check ---
        dist = np.linalg.norm(self.p1_pos - self.p2_pos)
        if not self.merged and dist < self.PIXEL_SIZE * 2:
            terminated = True
            self.game_over = True
            if self.steps >= self.MERGE_TIME_STEPS:
                # Successful merge
                self.merged = True
                self.merged_pos = (self.p1_pos + self.p2_pos) / 2
                self.merged_anim_timer = int(0.5 * self.FPS) # 0.5 second animation
                reward = 50.0  # Combined reward for surviving and merging
            else:
                # Collision
                self.collision_flash_timer = int(0.5 * self.FPS) # 0.5 second flash
                reward = -10.0

        # --- Max Steps Termination ---
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            truncated = True
            self.game_over = True
        
        # --- Survival Reward ---
        if not terminated:
            reward = 0.1

        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        self._render_particles()
        self._render_action_effects()

        if self.merged:
            # Animate the merged pixel
            size_mult = 1.5 + 0.5 * math.sin((self.merged_anim_timer / (0.5 * self.FPS)) * math.pi)
            if self.merged_anim_timer > 0:
                self.merged_anim_timer -= 1
            self._draw_pixel(self.merged_pos, self.PIXEL_SIZE * 2 * size_mult, self.COLOR_MERGED)
        elif not self.game_over or self.collision_flash_timer > 0:
            # Draw the two main pixels
            self._draw_pixel(self.p1_pos, self.PIXEL_SIZE, self.COLOR_P1)
            self._draw_pixel(self.p2_pos, self.PIXEL_SIZE, self.COLOR_P2)

        if self.collision_flash_timer > 0:
            self.collision_flash_timer -= 1
            alpha = int(200 * (self.collision_flash_timer / (0.5 * self.FPS)))
            flash_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surf.fill((*self.COLOR_COLLISION, alpha))
            self.screen.blit(flash_surf, (0, 0))

    def _render_ui(self):
        time_left = max(0, self.MERGE_TIME_SECONDS - (self.steps / self.FPS))
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI)
        self.screen.blit(timer_text, (10, 10))

        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            if self.merged:
                msg_text = self.font_msg.render("MERGED", True, self.COLOR_MERGED)
            elif self.steps >= self.MAX_STEPS: # Timed out
                msg_text = self.font_msg.render("TIME OUT", True, self.COLOR_UI)
            else: # Collision
                msg_text = self.font_msg.render("COLLISION", True, self.COLOR_COLLISION)
            
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

    def _draw_pixel(self, pos, radius, color):
        x, y = int(pos[0]), int(pos[1])
        r = int(max(1, radius))

        # Glow effect using multiple transparent circles
        glow_radius = int(r * 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        for i in range(glow_radius, 0, -glow_radius // 10 or 1):
            alpha = int(30 * (1 - (i / glow_radius))**2)
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, i, (*color, alpha))
        self.screen.blit(glow_surf, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Main pixel body
        pygame.gfxdraw.aacircle(self.screen, x, y, r, color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, r, color)

    def _add_particle(self, pos, color):
        p_pos = pos + self.np_random.uniform(-2, 2, size=2)
        p_vel = self.np_random.uniform(-1, 1, size=2)
        p_life = self.np_random.integers(15, 25)
        p_radius = self.np_random.uniform(2, 4)
        self.particles.append({'pos': p_pos, 'vel': p_vel, 'life': p_life, 'max_life': p_life, 'radius': p_radius, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            radius = int(p['radius'] * (p['life'] / p['max_life']))
            if radius > 0:
                particle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (radius, radius), radius)
                self.screen.blit(particle_surf, (int(p['pos'][0] - radius), int(p['pos'][1] - radius)), special_flags=pygame.BLEND_RGBA_ADD)

    def _add_action_effect(self, color):
        life = int(0.3 * self.FPS)
        self.action_effects.append({
            'pos': [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2],
            'radius': 0, 'max_radius': 80, 'life': life, 'max_life': life, 'color': color
        })

    def _update_action_effects(self):
        for e in self.action_effects:
            e['life'] -= 1
            e['radius'] = e['max_radius'] * (1 - (e['life'] / e['max_life']))
        self.action_effects = [e for e in self.action_effects if e['life'] > 0]
    
    def _render_action_effects(self):
        for e in self.action_effects:
            alpha = int(255 * (e['life'] / e['max_life']))
            if alpha > 0 and e['radius'] > 0:
                pygame.gfxdraw.aacircle(self.screen, int(e['pos'][0]), int(e['pos'][1]), int(e['radius']), (*e['color'], alpha))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Script ---
    # This part of the code is for testing and will not be used by the evaluation system.
    # It requires a display.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    game_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Fusion")
    clock = pygame.time.Clock()

    total_reward = 0
    terminated = False
    truncated = False
    
    print("\n" + "="*30)
    print(f"GAME: {env.__class__.__name__}")
    print(f"DESCRIPTION: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")
    print("R: Reset environment")
    print("Q: Quit")
    print("----------------\n")

    while running:
        action = np.array([0, 0, 0])  # Default action: no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
                    truncated = False
                    print(f"Environment reset. Initial score: {info['score']:.1f}")

        if not (terminated or truncated):
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"Episode finished! Final Score: {info['score']:.1f}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()