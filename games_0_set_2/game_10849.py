import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:05:18.311826
# Source Brief: brief_00849.md
# Brief Index: 849
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a neon-themed, rhythm-based sumo wrestling game.

    Two wrestlers battle in a circular arena. Players can teleport between four
    portals by successfully completing a short, randomly generated rhythm sequence.
    The goal is to push the opponent out of the ring by colliding with them while
    being closer to the center of the arena.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
      - In 'IDLE' state, this selects a target portal.
      - In 'RHYTHM' state, this is the input for the rhythm sequence.
    - actions[1]: Space button (0=released, 1=held)
      - A press (release -> held) initiates a teleport attempt to the selected portal.
    - actions[2]: Shift button (0=released, 1=held)
      - Currently unused.

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +100: Pushing the opponent out of the ring (winning).
    - -100: Being pushed out of the ring (losing).
    - +1: Pushing the opponent during a collision.
    - -1: Being pushed by the opponent during a collision.
    - +0.1: Successfully completing a teleport sequence.
    - -0.1: Failing a teleport sequence.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Push your opponent out of the neon ring in this rhythm-sumo battle. "
        "Complete rhythm sequences to teleport to strategic locations and gain the upper hand."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to select a portal. Press space to start a teleport, "
        "then match the on-screen arrow sequence to succeed. Push your opponent out of the ring!"
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    CENTER = (WIDTH // 2, HEIGHT // 2)
    RING_RADIUS = 160
    PORTAL_RING_RADIUS = 120
    PLAYER_RADIUS = 15
    PUSH_FORCE = 8
    RHYTHM_SEQUENCE_LENGTH = 3
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (10, 20, 30)
    COLOR_RING = (50, 80, 100)
    COLOR_P1 = (0, 150, 255)
    COLOR_P2 = (255, 50, 50)
    COLOR_PORTAL_INACTIVE = (0, 100, 50)
    COLOR_PORTAL_ACTIVE = (200, 255, 220)
    COLOR_P1_SELECT = (100, 200, 255)
    COLOR_P2_SELECT = (255, 120, 120)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SUCCESS = (100, 255, 100)
    COLOR_FAIL = (255, 100, 100)
    
    # Game States
    STATE_IDLE = 0
    STATE_RHYTHM = 1
    STATE_STUNNED = 2

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
        try:
            self.font_small = pygame.font.SysFont("Consolas", 16)
            self.font_medium = pygame.font.SysFont("Consolas", 24)
            self.font_large = pygame.font.SysFont("Consolas", 48)
        except pygame.error:
            self.font_small = pygame.font.SysFont(None, 20)
            self.font_medium = pygame.font.SysFont(None, 30)
            self.font_large = pygame.font.SysFont(None, 60)

        self.portal_positions = self._calculate_portal_positions()
        
        # This will be initialized in reset()
        self.p1_pos = None
        self.p2_pos = None
        self.p1_state = None
        self.p2_state = None
        self.p1_rhythm_sequence = None
        self.p1_rhythm_step = None
        self.p1_selected_portal = None
        self.p1_stun_timer = None
        self.p1_last_space_held = None
        self.ai_target_portal = None
        self.ai_state_timer = None
        self.ai_rhythm_step = None
        self.ai_reaction_time = None
        self.ai_initial_reaction_time = 1.0  # In seconds
        self.particles = []
        self.visual_effects = []
        self.steps = 0
        self.score = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        
        # Player 1 (Agent) State
        self.p1_pos = np.array([self.CENTER[0] - 50, self.CENTER[1]], dtype=float)
        self.p1_state = self.STATE_IDLE
        self.p1_rhythm_sequence = []
        self.p1_rhythm_step = 0
        self.p1_selected_portal = 0  # Corresponds to portal_positions[0]
        self.p1_stun_timer = 0
        self.p1_last_space_held = False

        # Player 2 (AI) State
        self.p2_pos = np.array([self.CENTER[0] + 50, self.CENTER[1]], dtype=float)
        self.p2_state = self.STATE_IDLE
        self.ai_target_portal = 0
        self.ai_state_timer = 0
        self.ai_rhythm_step = 0
        self.ai_reaction_time = 30 # steps, approx 1 sec at 30fps
        self.ai_difficulty_tier = 0

        # Effects
        self.particles = []
        self.visual_effects = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0.0

        # --- Handle Input and State Transitions ---
        reward += self._handle_player_input(movement, space_held)
        
        # --- Update AI ---
        ai_teleport_reward = self._update_ai()
        reward += ai_teleport_reward
        
        # --- Update Physics and Collisions ---
        collision_reward = self._update_physics()
        reward += collision_reward

        # --- Update Timers and Game State ---
        self._update_timers()
        self._update_difficulty()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not truncated: # Game ended by ring-out
            dist_p1 = np.linalg.norm(self.p1_pos - self.CENTER)
            dist_p2 = np.linalg.norm(self.p2_pos - self.CENTER)
            if dist_p1 > self.RING_RADIUS:
                reward -= 100.0
            elif dist_p2 > self.RING_RADIUS:
                reward += 100.0

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _calculate_portal_positions(self):
        positions = []
        for i in range(4):
            angle = math.pi / 2 * i
            x = self.CENTER[0] + self.PORTAL_RING_RADIUS * math.cos(angle)
            y = self.CENTER[1] + self.PORTAL_RING_RADIUS * math.sin(angle)
            positions.append(np.array([x, y]))
        # Order: Right, Up, Left, Down for intuitive mapping
        return [positions[0], positions[3], positions[2], positions[1]]

    def _handle_player_input(self, movement, space_held):
        reward = 0.0
        space_pressed = space_held and not self.p1_last_space_held
        self.p1_last_space_held = space_held

        if self.p1_state == self.STATE_IDLE:
            if 1 <= movement <= 4:
                self.p1_selected_portal = movement - 1
            if space_pressed:
                self.p1_state = self.STATE_RHYTHM
                self.p1_rhythm_sequence = self._generate_rhythm_sequence()
                self.p1_rhythm_step = 0
                # Sound: Rhythm Start

        elif self.p1_state == self.STATE_RHYTHM:
            if movement != 0: # Only process actual inputs
                expected_move = self.p1_rhythm_sequence[self.p1_rhythm_step]
                if movement == expected_move:
                    self.p1_rhythm_step += 1
                    if self.p1_rhythm_step >= self.RHYTHM_SEQUENCE_LENGTH:
                        # --- SUCCESS ---
                        self.p1_pos = self.portal_positions[self.p1_selected_portal].copy()
                        reward += 0.1
                        self.p1_state = self.STATE_IDLE
                        self._add_visual_effect(self.p1_pos, self.COLOR_SUCCESS, 60, 20)
                        self._add_particles(self.p1_pos, self.COLOR_P1, 30)
                        # Sound: Teleport Success
                    else:
                        # Sound: Rhythm Beat Success
                        pass
                else: # Wrong input
                    # --- FAILURE ---
                    reward -= 0.1
                    self.p1_state = self.STATE_STUNNED
                    self.p1_stun_timer = 45 # 1.5 seconds at 30fps
                    self._add_visual_effect(self.p1_pos, self.COLOR_FAIL, 30, 15)
                    # Sound: Teleport Fail
        
        return reward

    def _update_ai(self):
        self.ai_state_timer -= 1
        if self.p2_state == self.STATE_IDLE and self.ai_state_timer <= 0:
            # Decide to teleport
            self.p2_state = self.STATE_RHYTHM
            self.ai_target_portal = self._ai_choose_portal()
            self.ai_rhythm_step = 0
            self.ai_state_timer = self.ai_reaction_time # time to "input" first beat

        elif self.p2_state == self.STATE_RHYTHM and self.ai_state_timer <= 0:
            self.ai_rhythm_step += 1
            if self.ai_rhythm_step >= self.RHYTHM_SEQUENCE_LENGTH:
                # AI Teleport Success
                self.p2_pos = self.portal_positions[self.ai_target_portal].copy()
                self.p2_state = self.STATE_IDLE
                self.ai_state_timer = self.ai_reaction_time * 2 # Cooldown before next action
                self._add_visual_effect(self.p2_pos, self.COLOR_SUCCESS, 60, 20)
                self._add_particles(self.p2_pos, self.COLOR_P2, 30)
                return 0.1 # AI gets a conceptual reward
            else:
                self.ai_state_timer = self.ai_reaction_time # Time for next beat
        
        return 0

    def _ai_choose_portal(self):
        # AI strategy: find portal that puts it between player and center
        center = np.array(self.CENTER)
        best_portal_idx = -1
        max_dist_from_player = -1

        for i, portal_pos in enumerate(self.portal_positions):
            # A simple heuristic: choose the portal furthest from the player
            dist = np.linalg.norm(self.p1_pos - portal_pos)
            if dist > max_dist_from_player:
                max_dist_from_player = dist
                best_portal_idx = i

        return best_portal_idx if best_portal_idx != -1 else self.np_random.integers(0, 4)

    def _update_physics(self):
        reward = 0.0
        dist_vec = self.p1_pos - self.p2_pos
        dist = np.linalg.norm(dist_vec)
        
        if dist > 0 and dist < self.PLAYER_RADIUS * 2:
            # Sound: Collision
            self._add_particles((self.p1_pos + self.p2_pos) / 2, self.COLOR_TEXT, 10)
            
            p1_dist_center = np.linalg.norm(self.p1_pos - self.CENTER)
            p2_dist_center = np.linalg.norm(self.p2_pos - self.CENTER)
            
            push_vec = dist_vec / dist
            
            if p1_dist_center < p2_dist_center: # P1 pushes P2
                self.p2_pos += push_vec * self.PUSH_FORCE
                reward += 1.0
            elif p2_dist_center < p1_dist_center: # P2 pushes P1
                self.p1_pos -= push_vec * self.PUSH_FORCE
                reward -= 1.0
            else: # Equal distance, push both slightly
                self.p1_pos += push_vec * self.PUSH_FORCE / 2
                self.p2_pos -= push_vec * self.PUSH_FORCE / 2

        return reward

    def _update_timers(self):
        if self.p1_state == self.STATE_STUNNED:
            self.p1_stun_timer -= 1
            if self.p1_stun_timer <= 0:
                self.p1_state = self.STATE_IDLE

    def _update_difficulty(self):
        # Opponent reaction time decreases every 200 steps
        new_tier = self.steps // 200
        if new_tier > self.ai_difficulty_tier:
            self.ai_difficulty_tier = new_tier
            new_time_s = max(0.2, self.ai_initial_reaction_time - new_tier * 0.05)
            self.ai_reaction_time = int(new_time_s * 30) # Convert seconds to steps

    def _check_termination(self):
        dist_p1 = np.linalg.norm(self.p1_pos - self.CENTER)
        dist_p2 = np.linalg.norm(self.p2_pos - self.CENTER)
        
        if dist_p1 > self.RING_RADIUS or dist_p2 > self.RING_RADIUS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "p1_state": self.p1_state,
            "p2_dist_from_center": np.linalg.norm(self.p2_pos - self.CENTER)
        }
    
    def _generate_rhythm_sequence(self):
        return [self.np_random.integers(1, 5) for _ in range(self.RHYTHM_SEQUENCE_LENGTH)]

    # --- Rendering Methods ---

    def _render_game(self):
        # Update and draw particles
        self._update_and_draw_particles()
        
        # Draw the ring
        self._draw_glowing_circle(self.screen, self.COLOR_RING, self.CENTER, self.RING_RADIUS, 10, filled=False)
        
        # Draw portals
        for i, pos in enumerate(self.portal_positions):
            color = self.COLOR_PORTAL_INACTIVE
            glow = 5
            if self.p1_state != self.STATE_RHYTHM and i == self.p1_selected_portal:
                color = self.COLOR_P1_SELECT
                glow = 10
            if self.p2_state == self.STATE_RHYTHM and i == self.ai_target_portal:
                color = self.COLOR_P2_SELECT
                glow = 10
            if self.p1_state == self.STATE_RHYTHM and i == self.p1_selected_portal:
                color = self.COLOR_PORTAL_ACTIVE
                glow = 15
            self._draw_glowing_circle(self.screen, color, pos, 8, glow)
        
        # Draw players
        self._draw_glowing_circle(self.screen, self.COLOR_P2, self.p2_pos, self.PLAYER_RADIUS, 15)
        self._draw_glowing_circle(self.screen, self.COLOR_P1, self.p1_pos, self.PLAYER_RADIUS, 15)
        
        # Draw visual effects
        self._update_and_draw_visual_effects()

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Steps
        steps_text = self.font_medium.render(f"Step: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        # Player 1 State and Rhythm UI
        if self.p1_state == self.STATE_RHYTHM:
            self._draw_rhythm_ui(self.p1_pos, self.p1_rhythm_sequence, self.p1_rhythm_step)
        elif self.p1_state == self.STATE_STUNNED:
            stun_text = self.font_medium.render("STUNNED", True, self.COLOR_FAIL)
            pos = (self.p1_pos[0] - stun_text.get_width() / 2, self.p1_pos[1] - self.PLAYER_RADIUS - 30)
            self.screen.blit(stun_text, pos)

    def _draw_rhythm_ui(self, player_pos, sequence, current_step):
        ui_pos = (player_pos[0] - 35, player_pos[1] - self.PLAYER_RADIUS - 35)
        arrow_map = {1: "↑", 2: "↓", 3: "←", 4: "→"}
        
        for i, move in enumerate(sequence):
            color = self.COLOR_TEXT
            if i < current_step:
                color = self.COLOR_SUCCESS
            elif i == current_step:
                color = self.COLOR_PORTAL_ACTIVE
            
            arrow_text = self.font_large.render(arrow_map[move], True, color)
            self.screen.blit(arrow_text, (ui_pos[0] + i * 25, ui_pos[1]))

    def _draw_glowing_circle(self, surface, color, pos, radius, glow_strength, filled=True):
        pos_int = (int(pos[0]), int(pos[1]))
        for i in range(glow_strength, 0, -1):
            alpha = int(150 * (1 - i / glow_strength))
            glow_color = (*color, alpha)
            # Using pygame.draw as gfxdraw doesn't support alpha for filled shapes well
            temp_surf = pygame.Surface((radius * 2 + i * 2, radius * 2 + i * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (temp_surf.get_width() // 2, temp_surf.get_height() // 2), radius + i)
            surface.blit(temp_surf, (pos_int[0] - temp_surf.get_width() // 2, pos_int[1] - temp_surf.get_height() // 2))

        if filled:
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(radius), color)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], int(radius), color)

    def _add_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 30))
                pygame.draw.circle(self.screen, (*p['color'], alpha), [int(x) for x in p['pos']], int(p['life']/10 + 1))

    def _add_visual_effect(self, pos, color, max_radius, life):
        self.visual_effects.append({'pos': pos, 'color': color, 'max_radius': max_radius, 'life': life, 'max_life': life})

    def _update_and_draw_visual_effects(self):
        for fx in self.visual_effects[:]:
            fx['life'] -= 1
            if fx['life'] <= 0:
                self.visual_effects.remove(fx)
            else:
                progress = 1 - (fx['life'] / fx['max_life'])
                current_radius = int(fx['max_radius'] * progress)
                alpha = int(200 * (1 - progress))
                color = (*fx['color'], alpha)
                pos_int = (int(fx['pos'][0]), int(fx['pos'][1]))

                temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (current_radius, current_radius), current_radius, width=max(1, int(8 * (1 - progress))))
                self.screen.blit(temp_surf, (pos_int[0] - current_radius, pos_int[1] - current_radius))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    truncated = False
    
    # Pygame setup for manual play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Rhythm Sumo")
    clock = pygame.time.Clock()
    
    movement = 0
    space_held = 0
    shift_held = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset Environment")
    print("----------------------\n")

    while running:
        if terminated or truncated:
            end_reason = "Time limit reached" if truncated else "Ring out"
            print(f"Episode finished. Reason: {end_reason}. Final Score: {info['score']:.1f}. Press 'R' to reset.")
            while True:
                should_reset = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        should_reset = True
                        break
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        terminated = False
                        truncated = False
                        should_reset = True
                        break
                if should_reset:
                    break
            if not running:
                break

        # --- Get human input ---
        movement = 0
        space_held_current = (pygame.key.get_pressed()[pygame.K_SPACE] or pygame.key.get_pressed()[pygame.K_z])
        shift_held_current = pygame.key.get_pressed()[pygame.K_LSHIFT]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                # Use keydown for single-press actions like rhythm input
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4

        # If no new movement key is pressed, check held keys for portal selection
        if movement == 0:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

        action = [movement, 1 if space_held_current else 0, 1 if shift_held_current else 0]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()