import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:01:36.911992
# Source Brief: brief_00828.md
# Brief Index: 828
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player pulls levers to trigger probabilistic
    chain reactions. The goal is to complete 5 successful chains within 90 seconds.

    The environment is designed with a focus on visual quality and game feel, featuring
    smooth animations, particle effects, and clear UI feedback.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Pull levers to trigger probabilistic chain reactions. "
        "Complete 5 successful chains before time runs out to win."
    )
    user_guide = (
        "Controls: Use ← and → to select a lever. Press space to pull the selected lever."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 90
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Game Logic
    WIN_SCORE = 5
    NUM_LEVERS = 5
    CHAIN_STAGES = 3
    BASE_SUCCESS_PROB = 0.25
    PROB_DECAY = 0.05
    LEVER_ANIMATION_FRAMES = 45 # 0.75 seconds
    LEVER_COOLDOWN_FRAMES = 15 # 0.25 seconds

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_TIMER_BAR = (40, 160, 240)
    COLOR_TIMER_WARN = (250, 170, 50)
    COLOR_TIMER_DANGER = (220, 50, 50)
    
    COLOR_LEVER_BASE_IDLE = (60, 70, 80)
    COLOR_LEVER_HANDLE_IDLE = (120, 130, 140)
    COLOR_LEVER_SELECTED = (255, 255, 0)
    
    COLOR_STAGE_BOX_PENDING = (50, 60, 70)
    COLOR_STAGE_BOX_ACTIVE = (240, 200, 80)
    COLOR_STAGE_SUCCESS = (50, 220, 100)
    COLOR_STAGE_FAIL = (220, 50, 50)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20)
        self.font_title = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.levers = []
        self.particles = []
        self.selected_lever_index = 0
        self.last_space_held = False
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        # self.validate_implementation() # Removed for submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_lever_index = 2
        self.last_space_held = False
        self.particles = []
        
        self.levers = []
        for i in range(self.NUM_LEVERS):
            self.levers.append({
                "state": "idle", # idle, active, cooldown
                "animation_timer": 0,
                "cooldown_timer": 0,
                "stage_results": ["pending"] * self.CHAIN_STAGES,
                "stage_probs": [self.BASE_SUCCESS_PROB] * self.CHAIN_STAGES
            })
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0

        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        
        reward += self._handle_input(action)
        self._update_game_state()
        
        terminated = self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS
        truncated = False # No truncation condition other than termination
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 10.0 # Win bonus
                # SFX: Game Win
            else:
                reward -= 10.0 # Lose penalty
                # SFX: Game Lose
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_action, _ = action
        
        # --- Handle Cursor Movement ---
        # Action '0' is no-op, '1' and '2' are unused for movement
        if movement == 3: # Left
            self.selected_lever_index = (self.selected_lever_index - 1 + self.NUM_LEVERS) % self.NUM_LEVERS
        elif movement == 4: # Right
            self.selected_lever_index = (self.selected_lever_index + 1) % self.NUM_LEVERS

        # --- Handle Lever Pull ---
        space_held = space_action == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held
        
        if space_pressed:
            lever = self.levers[self.selected_lever_index]
            if lever["state"] == "idle":
                return self._pull_lever(self.selected_lever_index)
        return 0.0

    def _pull_lever(self, index):
        # SFX: Lever Pull
        lever = self.levers[index]
        lever["state"] = "active"
        lever["animation_timer"] = self.LEVER_ANIMATION_FRAMES
        lever["stage_results"] = ["pending"] * self.CHAIN_STAGES
        lever["stage_probs"] = [self.BASE_SUCCESS_PROB] * self.CHAIN_STAGES

        reward = 0.0
        current_probs = list(lever["stage_probs"])
        is_full_chain = True

        for i in range(self.CHAIN_STAGES):
            if self.np_random.random() < current_probs[i]:
                lever["stage_results"][i] = "success"
                reward += 0.1
                # SFX: Stage Success
            else:
                lever["stage_results"][i] = "fail"
                is_full_chain = False
                # SFX: Stage Fail
                # Reduce probability for subsequent stages
                for j in range(i + 1, self.CHAIN_STAGES):
                    current_probs[j] = max(0, current_probs[j] - self.PROB_DECAY)
        
        if is_full_chain:
            reward += 1.0
            self.score += 1
            # SFX: Full Chain Success
            self._spawn_particles(index, self.COLOR_STAGE_SUCCESS, 50)
        else:
            self._spawn_particles(index, self.COLOR_STAGE_FAIL, 20)
            
        return reward

    def _update_game_state(self):
        # Update Levers
        for lever in self.levers:
            if lever["state"] == "active":
                lever["animation_timer"] -= 1
                if lever["animation_timer"] <= 0:
                    lever["state"] = "cooldown"
                    lever["cooldown_timer"] = self.LEVER_COOLDOWN_FRAMES
            elif lever["state"] == "cooldown":
                lever["cooldown_timer"] -= 1
                if lever["cooldown_timer"] <= 0:
                    lever["state"] = "idle"
                    lever["stage_results"] = ["pending"] * self.CHAIN_STAGES

        # Update Particles
        new_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def _spawn_particles(self, lever_index, color, count):
        lever_x = (self.SCREEN_WIDTH / (self.NUM_LEVERS + 1)) * (lever_index + 1)
        spawn_y = self.SCREEN_HEIGHT / 2
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": [lever_x, spawn_y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(20, 40),
                "color": color,
                "radius": self.np_random.uniform(2, 5)
            })
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_particles()
        for i in range(self.NUM_LEVERS):
            self._render_lever(i)

    def _render_lever(self, index):
        lever = self.levers[index]
        is_selected = index == self.selected_lever_index
        
        lever_w = 80
        lever_h = 40
        lever_x = (self.SCREEN_WIDTH / (self.NUM_LEVERS + 1)) * (index + 1) - lever_w / 2
        lever_y = self.SCREEN_HEIGHT * 0.75

        # Draw Selection Highlight
        if is_selected and not self.game_over:
            glow_rect = pygame.Rect(lever_x - 5, lever_y - 5, lever_w + 10, lever_h + 10)
            pygame.draw.rect(self.screen, self.COLOR_LEVER_SELECTED, glow_rect, 3, border_radius=8)

        # Draw Lever Base
        base_rect = pygame.Rect(lever_x, lever_y, lever_w, lever_h)
        pygame.draw.rect(self.screen, self.COLOR_LEVER_BASE_IDLE, base_rect, border_radius=5)
        
        # Draw Lever Handle
        handle_x = lever_x + lever_w / 2
        handle_y_offset = 0
        if lever["state"] == "active":
            progress = 1 - (lever["animation_timer"] / self.LEVER_ANIMATION_FRAMES)
            handle_y_offset = math.sin(progress * math.pi) * -10
        pygame.draw.circle(self.screen, self.COLOR_LEVER_HANDLE_IDLE, (int(handle_x), int(lever_y + handle_y_offset)), 10)

        # Draw Chain Reaction Display
        stage_area_y = self.SCREEN_HEIGHT / 2 - 50
        stage_w = 20
        stage_h = 40
        total_stage_w = self.CHAIN_STAGES * stage_w + (self.CHAIN_STAGES - 1) * 10
        start_x = lever_x + (lever_w - total_stage_w) / 2

        for i in range(self.CHAIN_STAGES):
            stage_x = start_x + i * (stage_w + 10)
            stage_rect = pygame.Rect(stage_x, stage_area_y, stage_w, stage_h)
            
            color = self.COLOR_STAGE_BOX_PENDING
            if lever["state"] == "active":
                anim_progress = 1.0 - (lever["animation_timer"] / self.LEVER_ANIMATION_FRAMES)
                stage_start_time = i / self.CHAIN_STAGES
                stage_end_time = (i + 1) / self.CHAIN_STAGES

                if stage_start_time <= anim_progress < stage_end_time:
                    color = self.COLOR_STAGE_BOX_ACTIVE
                elif anim_progress >= stage_end_time:
                    if lever["stage_results"][i] == "success":
                        color = self.COLOR_STAGE_SUCCESS
                    else:
                        color = self.COLOR_STAGE_FAIL
            
            pygame.draw.rect(self.screen, color, stage_rect, border_radius=4)
            pygame.draw.rect(self.screen, tuple(min(255, c+30) for c in color), stage_rect, 2, border_radius=4)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, int(255 * (p["life"] / 40.0)))
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((int(p["radius"])*2, int(p["radius"])*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(
                temp_surf, int(p["radius"]), int(p["radius"]), int(p["radius"]), p["color"] + (alpha,)
            )
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])))


    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"CHAINS: {self.score} / {self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer Bar
        time_ratio = 1.0 - (self.steps / self.MAX_STEPS)
        bar_width = (self.SCREEN_WIDTH - 20) * time_ratio
        bar_color = self.COLOR_TIMER_BAR
        if time_ratio < 0.5: bar_color = self.COLOR_TIMER_WARN
        if time_ratio < 0.2: bar_color = self.COLOR_TIMER_DANGER
        
        pygame.draw.rect(self.screen, self.COLOR_LEVER_BASE_IDLE, (10, self.SCREEN_HEIGHT - 30, self.SCREEN_WIDTH - 20, 20))
        if bar_width > 0:
            pygame.draw.rect(self.screen, bar_color, (10, self.SCREEN_HEIGHT - 30, int(bar_width), 20))

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.score >= self.WIN_SCORE:
                end_text = self.font_title.render("YOU WIN!", True, self.COLOR_STAGE_SUCCESS)
            else:
                end_text = self.font_title.render("TIME'S UP", True, self.COLOR_STAGE_FAIL)
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually.
    # It will not be executed by the autograder.
    
    # Un-dummy the video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    done = False
    
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Lever Chain Reaction")
    
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # [movement, space, shift]
    
    while not done:
        # --- Player Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        action[0] = 0 # No movement by default
        
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}")

        # --- Render to Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']}")
    
    pygame.time.wait(3000)
    
    env.close()